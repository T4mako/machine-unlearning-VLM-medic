import math
import time
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from config import config
from model.model_wrapper import GenerativeQwenVLModel


class KGATrainer:
    def __init__(self, A_star: GenerativeQwenVLModel,
                 retain_data: List[Dict], forget_data: List[Dict], dn_data: List[Dict], val_data: List[Dict],
                 lr=5e-6, batch_size=2, log_interval=5, debug_limit=None,
                 load_AD: bool = True, load_Af: bool = True, load_An: bool = True,
                 baseline_gap_override: Optional[float] = None,
                 af_nll_path: Optional[str] = None):
        self.A_star = A_star  # 待遗忘模型，初始化自 AD 权重
        
        # 根据配置选择优化参数（是否只训练遗忘层）
        if bool(getattr(config.train, "freeze_backbone", True)) and len(self.A_star.get_unlearning_parameters()) > 0:
            # 冻结主干，仅训练遗忘层
            for p in self.A_star.model.parameters():
                p.requires_grad = False
            opt_params = list(self.A_star.get_unlearning_parameters())
        else:
            opt_params = list(self.A_star.parameters())
        self.optimizer = torch.optim.AdamW(opt_params, lr=lr)

        self.retain_data = retain_data  # Dr
        self.forget_data = forget_data  # Df
        self.dn_data = dn_data          # Dn
        self.val_data = val_data
        self.batch_size = int(batch_size)
        self.log_interval = int(log_interval)
        self.debug_limit = int(debug_limit) if debug_limit is not None else None

        # 从配置读取KGA/EUL超参
        self.alpha = float(config.kga.alpha)
        self.sigma = float(config.kga.sigma)
        self.use_nll_gap = bool(config.kga.use_nll_gap)
        self.objective = str(getattr(config.train, "objective", "kga")).lower()

        # 选择性构建 AD、Af、An（用于教师/基线度量）
        self.AD = self._load_model(config.kga.ad_checkpoint) if load_AD else None
        self.Af = self._load_model(config.kga.af_checkpoint) if load_Af else None
        self.An = self._load_model(config.kga.an_checkpoint) if load_An else None

        # 载入Af的batch级NLL缓存（可选），用于KGA时免加载Af模型
        self.af_nll_batches = None
        if af_nll_path is not None:
            try:
                payload = torch.load(af_nll_path, map_location="cpu")
                if isinstance(payload, dict):
                    if "batch_nll" in payload:
                        nll = payload["batch_nll"]
                    elif "nll" in payload:  # 兼容逐样本格式
                        nll = payload["nll"]
                    else:
                        nll = payload
                else:
                    nll = payload
                if hasattr(nll, "tolist"):
                    nll = nll.tolist()
                nll = [float(x) for x in nll]
                # 统一为逐batch的均值
                total_len = len(self.forget_data if self.debug_limit is None else self.forget_data[: self.debug_limit])
                import math as _math
                expected_batches = _math.ceil(total_len / max(self.batch_size, 1))
                if len(nll) == total_len:
                    # 逐样本 -> 逐batch均值
                    batches = []
                    for i in range(0, total_len, self.batch_size):
                        chunk = nll[i:i + self.batch_size]
                        batches.append(sum(chunk) / max(len(chunk), 1))
                    self.af_nll_batches = batches
                    print(f"[KGA] Loaded Af NLL cache (per-sample) and aggregated to batches={len(self.af_nll_batches)} from {af_nll_path}")
                elif len(nll) == expected_batches:
                    self.af_nll_batches = nll
                    print(f"[KGA] Loaded Af NLL cache (per-batch) batches={len(self.af_nll_batches)} from {af_nll_path}")
                else:
                    self.af_nll_batches = nll
                    print(f"[KGA][WARN] Af NLL cache length={len(nll)} 与样本/批次数不匹配 (samples={total_len}, batches={expected_batches})，将按step索引使用，可能不完全对齐。")
            except Exception as e:
                print(f"[KGA][WARN] Failed to load Af NLL cache from {af_nll_path}: {e}")

        # 预计算基线差距 G = KL[AD(z), An(z)] 在 Dn 上的均值（用NLL差近似）
        if baseline_gap_override is not None:
            self.baseline_gap = float(baseline_gap_override)
            print(f"[KGA] Using overridden baseline gap (G): {self.baseline_gap:.6f}")
        elif (self.AD is not None) and (self.An is not None):
            self.baseline_gap = self._estimate_baseline_gap(self.AD, self.An, self.dn_data)
            print(f"[KGA] Baseline gap (G) on Dn: {self.baseline_gap:.6f}")
        else:
            # 当未加载AD或An时，无法计算baseline_gap；设为0并提示。
            self.baseline_gap = 0.0
            print("[KGA][WARN] AD/An not both loaded, baseline_gap set to 0. Consider providing baseline_gap_override or enable load_AD/load_An.")

    def _load_model(self, ckpt_path):
        # 若提供checkpoint可在此load; 否则构造同权重模型
        model = GenerativeQwenVLModel(model_name=config.model.model_name, use_fast=config.model.use_fast)
        # 禁用教师/基线模型的遗忘层，确保其行为稳定
        try:
            model.enable_unlearning(False)
        except Exception:
            pass
        if ckpt_path:
            state = torch.load(ckpt_path, map_location=model.device)
            model.load_state_dict(state)
        return model

    def _iter_batches(self, data: List[Dict]):
        dataset = data if self.debug_limit is None else data[: self.debug_limit]
        B = self.batch_size
        for i in range(0, len(dataset), B):
            batch_items = dataset[i: i + B]
            images = [x["image"] for x in batch_items]
            texts = [x["text"] for x in batch_items]
            targets = [x["target"] for x in batch_items]
            yield images, texts, targets, i // B, __import__('math').ceil(len(dataset) / B)

    @torch.no_grad()
    def _estimate_baseline_gap(self, M1: Optional[GenerativeQwenVLModel], M2: Optional[GenerativeQwenVLModel], data: List[Dict]):
        if (M1 is None) or (M2 is None):
            raise RuntimeError("_estimate_baseline_gap requires both models loaded (AD and An).")
        # 用 NLL 差近似 KL 差：E_z[ NLL_M1(z) - NLL_M2(z) ] 的绝对值
        total = 0.0
        count = 0
        for images, texts, targets, step, total_steps in self._iter_batches(data):
            nll1 = float(M1.compute_nll(images, texts, targets).item())
            nll2 = float(M2.compute_nll(images, texts, targets).item())
            total += abs(nll1 - nll2)
            count += 1
        return total / max(count, 1)

    def train(self, epochs=3):
        # 目标与依赖检查
        if self.objective == "eul":
            if self.alpha > 0 and self.AD is None:
                raise RuntimeError("EUL objective requires AD (teacher) to be loaded. Please set load_AD=True or set alpha=0.")
        else:  # KGA
            if (self.Af is None) and (self.af_nll_batches is None):
                raise RuntimeError("KGA objective requires Af or Af-NLL cache. Please set load_Af=True or provide af_nll_path cache.")
            if (self.AD is None) or (self.An is None):
                print("[KGA][WARN] Training without AD/An loaded; baseline_gap will remain as set (default 0.0). Consider enabling load_AD/load_An or provide baseline_gap_override.")

        # 初始知识差距 G（用 AD 与 Af 在 Df 上的差距）
        with torch.no_grad():
            if (self.AD is not None) and (self.Af is not None):
                init_gap = self._estimate_baseline_gap(self.AD, self.Af, self.forget_data)
                print(f"[KGA] Initial gap on Df between AD and Af: {init_gap:.6f}")
            else:
                print("[KGA] Skipping initial AD-Af gap computation (teacher/reference not fully loaded or using cache).")

        target_threshold = self.sigma * max(self.baseline_gap, 1e-8)
        print(f"[KGA] Early-stop threshold: {target_threshold:.6f}")

        global_step = 0
        for epoch in range(epochs):
            print(f"[KGA] Epoch {epoch+1}/{epochs}")
            total_loss = 0.0
            total_steps = 0
            t0 = time.time()

            for images, texts, targets, step, steps_total in self._iter_batches(self.forget_data):
                self.A_star.train()
                self.optimizer.zero_grad()

                if self.objective == "eul":
                    # EUL：在 Df 上增大损失，同时在 Dr 上与 AD 保持一致
                    out_f = self.A_star.forward(images, texts, targets)
                    nll_astar_f = out_f.loss  # 需要梯度
                    L_forget = - nll_astar_f

                    # 保持项：在 Dr 上接近 AD
                    if (len(self.retain_data) > 0) and (self.AD is not None):
                        ridx = (global_step + step) % max(len(self.retain_data), 1)
                        r_end = min(ridx + self.batch_size, len(self.retain_data))
                        r_batch = self.retain_data[ridx:r_end]
                        r_images = [x["image"] for x in r_batch]
                        r_texts = [x["text"] for x in r_batch]
                        r_targets = [x["target"] for x in r_batch]
                        out_r = self.A_star.forward(r_images, r_texts, r_targets)
                        nll_astar_r = out_r.loss  # 需要梯度
                        with torch.no_grad():
                            nll_ad_r = self.AD.compute_nll(r_images, r_texts, r_targets)
                        L_retain = torch.abs(nll_astar_r - nll_ad_r)
                    else:
                        L_retain = torch.tensor(0.0, device=self.A_star.device)

                    loss = L_forget + self.alpha * L_retain

                else:
                    # KGA：让 A* 与 Af 在 Df 的gap 接近 AD 与 An 在 Dn 的gap
                    out_f = self.A_star.forward(images, texts, targets)
                    nll_astar = out_f.loss  # 需要梯度
                    if self.Af is not None:
                        with torch.no_grad():
                            nll_af = self.Af.compute_nll(images, texts, targets)
                    else:
                        if (self.af_nll_batches is None) or (step >= len(self.af_nll_batches)):
                            raise RuntimeError("Af NLL cache不可用或越界，请先运行 --mode precompute_af_nll 并确保batch_size/debug_limit一致。")
                        nll_af = torch.tensor(self.af_nll_batches[step], device=self.A_star.device, dtype=nll_astar.dtype)
                    gap_star = torch.abs(nll_astar - nll_af)

                    # 基线 gap_base = | NLL_AD(Dn) - NLL_An(Dn) | (用预估均值代替逐batch)
                    gap_base = torch.tensor(self.baseline_gap, device=self.A_star.device, dtype=gap_star.dtype)
                    La = torch.abs(gap_star - gap_base)

                    # 性能保持 Lr：在 Dr 上让 A* 接近 AD
                    if (len(self.retain_data) > 0) and (self.AD is not None):
                        ridx = (global_step + step) % max(len(self.retain_data), 1)
                        r_end = min(ridx + self.batch_size, len(self.retain_data))
                        r_batch = self.retain_data[ridx:r_end]
                        r_images = [x["image"] for x in r_batch]
                        r_texts = [x["text"] for x in r_batch]
                        r_targets = [x["target"] for x in r_batch]
                        out_r = self.A_star.forward(r_images, r_texts, r_targets)
                        nll_astar_r = out_r.loss  # 需要梯度
                        with torch.no_grad():
                            nll_ad_r = self.AD.compute_nll(r_images, r_texts, r_targets)
                        Lr = torch.abs(nll_astar_r - nll_ad_r)
                    else:
                        Lr = torch.tensor(0.0, device=self.A_star.device)

                    loss = La + self.alpha * Lr

                loss.backward()
                self.optimizer.step()

                total_loss += float(loss.item())
                total_steps += 1
                global_step += 1

                if (step + 1) % self.log_interval == 0 or (step + 1) == steps_total:
                    dt = time.time() - t0
                    ips = (total_steps * self.batch_size) / max(dt, 1e-6)
                    if self.objective == "eul":
                        print(f"[EUL][train] step {step+1}/{steps_total} | L_forget={float(L_forget.item()):.4f} L_retain={float(L_retain.item()):.4f} loss={float(loss.item()):.4f} | {ips:.1f} samples/s")
                    else:
                        print(f"[KGA][train] step {step+1}/{steps_total} | loss={float(loss.item()):.4f} | {ips:.1f} samples/s")

            epoch_loss = total_loss / max(total_steps, 1)
            print(f"[{self.objective.upper()}] Epoch {epoch+1} avg loss: {epoch_loss:.6f}")

            # 监控当前在 Df 上的gap_* 均值，用于早停（仅 KGA 有明确gap对齐目标）
            if self.objective != "eul":
                with torch.no_grad():
                    current_gap = 0.0
                    cnt = 0
                    for images, texts, targets, step, steps_total in self._iter_batches(self.forget_data):
                        nll_astar = float(self.A_star.compute_nll(images, texts, targets).item())
                        if self.Af is not None:
                            nll_af = float(self.Af.compute_nll(images, texts, targets).item())
                        else:
                            if (self.af_nll_batches is None) or (step >= len(self.af_nll_batches)):
                                raise RuntimeError("Af NLL cache不可用或越界，请先运行 --mode precompute_af_nll 并确保batch_size/debug_limit一致。")
                            nll_af = float(self.af_nll_batches[step])
                        current_gap += abs(nll_astar - nll_af)
                        cnt += 1
                    current_gap /= max(cnt, 1)
                print(f"[KGA] Current gap_* on Df: {current_gap:.6f}")

                if current_gap <= (self.sigma * max(self.baseline_gap, 1e-8)):
                    print(f"[KGA] Early stop: gap_* ({current_gap:.6f}) <= threshold ({self.sigma * max(self.baseline_gap, 1e-8):.6f})")
                    break

        print("[KGA] Training finished.")