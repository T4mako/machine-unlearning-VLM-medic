import logging
import math
import time
import os
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from config import config
from model.model_wrapper import GenerativeQwenVLModel, auto_load_lora_or_pt
from torchvision.transforms import ToTensor

# 可选：PEFT 适配器加载
try:
    from peft import PeftModel  # type: ignore
except Exception:
    PeftModel = None  # type: ignore


# 可选：bitsandbytes 8-bit 优化器
try:
    import bitsandbytes as bnb  # type: ignore
except Exception:
    bnb = None  # type: ignore

from torch.amp import autocast as _autocast, GradScaler  # torch>=2.0


to_tensor = ToTensor()


class KGATrainer:
    def __init__(self, A_star: GenerativeQwenVLModel,
                 retain_data: List[Dict], forget_data: List[Dict], dn_data: List[Dict], val_data: List[Dict],
                 lr=config.train.lr, 
                 batch_size=config.train.batch_size, 
                 log_interval=config.train.log_interval, 
                 debug_limit=config.train.debug_limit,
                 load_AD: bool = False, 
                 load_Af: bool = True, 
                 load_An: bool = False,
                 baseline_gap_override: Optional[float] = None,
                 af_nll_path: Optional[str] = None):
        self.A_star = A_star  # 待遗忘模型，初始化自 AD 权重
        
        # 根据 requires_grad 自动选择可训练参数（兼容LoRA与遗忘层开关）
        opt_params = [p for p in self.A_star.parameters() if getattr(p, 'requires_grad', False)]
        trainable_count = sum(p.numel() for p in opt_params)
        total_params = sum(p.numel() for p in self.A_star.parameters())
        logging.info(f"[Trainer] 可训练参数张量数: {len(opt_params)} | 元素总数: {trainable_count}/{total_params}")

        # 训练精度与AMP
        self.precision = str(getattr(config.model, "precision", "bf16")).lower()
        self._amp_enabled = (torch.cuda.is_available() and (_autocast is not None) and (self.precision in ["bf16", "fp16"]))
        self._amp_dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(self._amp_enabled and self.precision == "fp16")) if GradScaler is not None else None

        # 梯度累积
        self.grad_accum_steps = max(1, int(getattr(config.train, "gradient_accumulation_steps", 1)))

        # 选择优化器：可选8-bit优化器
        use_8bit_opt = bool(getattr(config.train, "use_8bit_optimizer", False))
        if use_8bit_opt and (bnb is not None):
            try:
                self.optimizer = bnb.optim.AdamW8bit(opt_params, lr=lr)
                print("[Trainer] Using bitsandbytes AdamW8bit optimizer.")
            except Exception as e:
                print(f"[Trainer][WARN] Failed to init AdamW8bit: {e}. Falling back to torch.optim.AdamW.")
                self.optimizer = torch.optim.AdamW(opt_params, lr=lr)
        else:
            self.optimizer = torch.optim.AdamW(opt_params, lr=lr)

        self.retain_data = retain_data  # Dr
        self.forget_data = forget_data  # Df
        self.dn_data = dn_data          # Dn
        self.val_data = val_data
        self.batch_size = int(batch_size)
        self.log_interval = int(log_interval)
        self.debug_limit = int(debug_limit) if debug_limit is not None else None

        # 超参（融合）
        self.alpha = float(config.kga.alpha)      # 保持项权重 (Lr)
        self.sigma = float(config.kga.sigma)      # 对齐误差相对阈值
        self.use_nll_gap = bool(config.kga.use_nll_gap)
        self.objective = "fusion"                 # 统一为融合目标
        self.lambda_f = float(getattr(config.kga, "lambda_f", 1.0))  # （保留以向后兼容，未使用）
        self.beta = float(getattr(config.kga, "beta", 1.0))          # 对齐项权重

        # 选择性构建 AD、Af、An（用于教师/基线度量）
        self.AD = self._load_model(config.kga.ad_checkpoint) if load_AD else None
        self.Af = self._load_model(config.kd.af_out_ckpt) if load_Af else None
        self.An = self._load_model(config.kd.an_out_ckpt) if load_An else None

        # 载入Af的batch级NLL缓存（可选），用于免加载Af模型
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
                    print(f"[Fusion] Loaded Af NLL cache (per-sample) and aggregated to batches={len(self.af_nll_batches)} from {af_nll_path}")
                elif len(nll) == expected_batches:
                    self.af_nll_batches = nll
                    print(f"[Fusion] Loaded Af NLL cache (per-batch) batches={len(self.af_nll_batches)} from {af_nll_path}")
                else:
                    self.af_nll_batches = nll
                    print(f"[Fusion][WARN] Af NLL cache length={len(nll)} 与样本/批次数不匹配 (samples={total_len}, batches={expected_batches})，将按step索引使用，可能不完全对齐。")
            except Exception as e:
                print(f"[Fusion][WARN] Failed to load Af NLL cache from {af_nll_path}: {e}")

        # 预计算基线差距 G = |NLL_AD(Dn) - NLL_An(Dn)| 的均值（用NLL差近似）
        if baseline_gap_override is not None:
            self.baseline_gap = float(baseline_gap_override)
            print(f"[Fusion] Using overridden baseline gap (G): {self.baseline_gap:.6f}")
        elif (self.AD is not None) and (self.An is not None):
            self.baseline_gap = self._estimate_baseline_gap(self.AD, self.An, self.dn_data)
            print(f"[Fusion] Baseline gap (G) on Dn: {self.baseline_gap:.6f}")
        else:
            # 当未加载AD或An时，无法计算baseline_gap；设为0并提示。
            self.baseline_gap = 0.0
            print("[Fusion][WARN] AD/An not both loaded, baseline_gap set to 0. Consider providing baseline_gap_override or enable load_AD/load_An.")

    def _load_model(self, ckpt_path):
        logging.info(f"[Trainer] Loading model from checkpoint: {ckpt_path}")
        # 选择基座模型名称：若是KD产物（LoRA适配器），则使用学生基座；否则使用全局大模型
        base_name = config.model.model_name
        try:
            if ckpt_path and os.path.abspath(str(ckpt_path)) in [
                os.path.abspath(config.kd.an_out_ckpt),
                os.path.abspath(config.kd.af_out_ckpt),
            ]:
                base_name = getattr(config.kd, 'student_model_name', None) or base_name
        except Exception:
            pass
        # 若提供checkpoint可在此load; 否则构造同权重模型
        model = GenerativeQwenVLModel(model_name=base_name, use_fast=config.model.use_fast, lora_enabled=False)
        # 禁用教师/基线模型的遗忘层，确保其行为稳定
        try:
            model.enable_unlearning(False)
        except Exception:
            pass
        if ckpt_path:
            try:
                # 统一从同名目录优先加载LoRA适配器，其次回退全量.pt
                model.model = auto_load_lora_or_pt(model.model, str(ckpt_path), device=model.device)
                logging.info(f"[Trainer] 已加载权重: {ckpt_path}")
            except Exception as e:
                logging.warning(f"[Trainer] 加载checkpoint失败 {ckpt_path}: {e}")
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
        # 用 NLL 差近似 KL 差：E_z[ |NLL_M1(z) - NLL_M2(z)| ]
        total = 0.0
        count = 0
        for images, texts, targets, step, total_steps in self._iter_batches(data):
            nll1 = float(M1.compute_nll(images, texts, targets).item())
            nll2 = float(M2.compute_nll(images, texts, targets).item())
            total += abs(nll1 - nll2)
            count += 1
        return total / max(count, 1)

    def train(self, epochs=3):
        # 融合目标依赖项检查：按权重启用
        gap_enabled = (self.beta > 0.0) and ((self.Af is not None) or (self.af_nll_batches is not None))
        if self.alpha > 0.0 and self.AD is None:
            print("[Fusion][WARN] alpha>0 但未加载AD，保持项将被跳过（L_r=0）。如需启用请设置 load_AD=True 或提供 AD 权重。")
        if (self.beta > 0.0) and not gap_enabled:
            print("[Fusion][WARN] beta>0 但未加载Af且未提供Af-NLL缓存，对齐项将被跳过（L_gap=0）。如需启用请设置 load_Af=True 或提供 af_nll_path。")
        if gap_enabled:
            if (self.An is None) and (self.baseline_gap == 0.0):
                print("[Fusion][WARN] 未加载An且未提供baseline_gap_override，baseline_gap将为0，早停阈值可能过低。建议提供baseline_gap_override或启用load_An计算。")

        # 初始对齐误差（可选）：在 Df 上度量 |La - G|
        with torch.no_grad():
            if gap_enabled and (self.Af is not None):
                init_align_err = 0.0
                cnt = 0
                for images, texts, targets, step, steps_total in self._iter_batches(self.forget_data):
                    nll_astar = float(self.A_star.compute_nll(images, texts, targets).item())
                    nll_af = float(self.Af.compute_nll(images, texts, targets).item())
                    La = abs(nll_astar - nll_af)
                    init_align_err += abs(La - float(self.baseline_gap))
                    cnt += 1
                init_align_err /= max(cnt, 1)
                print(f"[Fusion] Initial alignment error on Df: {init_align_err:.6f}")
            else:
                print("[Fusion] Skipping initial alignment error computation (disabled or missing Af/cache).")

        # 早停阈值：|mean(La) - G| <= sigma * |G|
        target_threshold = None
        if gap_enabled:
            target_threshold = self.sigma * max(abs(self.baseline_gap), 1e-8)
            print(f"[Fusion] Early-stop threshold (relative to G): {target_threshold:.6f}")
        else:
            print("[Fusion] Early-stop disabled (gap term not enabled).")

        global_step = 0
        self.optimizer.zero_grad()
        for epoch in range(epochs):
            print(f"[Fusion] Epoch {epoch+1}/{epochs}")
            # ===== 遗忘层参数日志（Epoch 前）=====
            if hasattr(self.A_star, "unlearning_layer") and (self.A_star.unlearning_layer is not None):
                if epoch == 0:
                    logging.info("====[UNL] 遗忘层已启用，开始记录每个epoch的参数变化====")
                # 保存本epoch开始时的参数快照，用于计算delta
                try:
                    self._unl_prev_state = {
                        name: p.detach().cpu().clone()
                        for name, p in self.A_star.unlearning_layer.named_parameters()
                    }
                except Exception:
                    self._unl_prev_state = {}
                # 输出统计信息
                for name, p in self.A_star.unlearning_layer.named_parameters():
                    data = p.detach()
                    mean_val = float(data.mean().item())
                    std_val = float(data.std(unbiased=False).item())
                    req = bool(p.requires_grad)
                    logging.info(f"====[UNL][epoch {epoch+1} pre] {name} shape={tuple(p.shape)} requires_grad={req} mean={mean_val:.6f} std={std_val:.6f}====")
            else:
                if epoch == 0:
                    logging.info("====[UNL] 遗忘层未启用，跳过参数日志====")

            total_loss = 0.0
            total_steps = 0
            t0 = time.time()

            accum = 0
            for images, texts, targets, step, steps_total in self._iter_batches(self.forget_data):
                # 训练模式
                self.A_star.train()

                # 前向与损失构造（AMP）
                with _autocast(device_type="cuda", dtype=self._amp_dtype):
                    # 1) Df 上的对齐项 La = |NLL(A*) - NLL(Af)|
                    out_f = self.A_star.forward(images, texts, targets)
                    nll_astar = out_f.loss  # 需要梯度
                    if gap_enabled:
                        if self.Af is not None:
                            with torch.no_grad():
                                nll_af = self.Af.compute_nll(images, texts, targets)
                        else:
                            if (self.af_nll_batches is None) or (step >= len(self.af_nll_batches)):
                                raise RuntimeError("Af NLL cache不可用或越界，请先运行 --mode precompute_af_nll 并确保batch_size/debug_limit一致。")
                            nll_af = torch.tensor(self.af_nll_batches[step], device=self.A_star.device, dtype=nll_astar.dtype)
                        La = torch.abs(nll_astar - nll_af)
                        gap_base = torch.tensor(self.baseline_gap, device=self.A_star.device, dtype=La.dtype)
                        L_gap = torch.abs(La - gap_base)
                    else:
                        # 若未启用gap项，则仍定义 La 作为忘记对齐（默认与 Af 距离），但 Af 缓存不可用时置0
                        try:
                            if self.Af is not None:
                                with torch.no_grad():
                                    nll_af = self.Af.compute_nll(images, texts, targets)
                                La = torch.abs(nll_astar - nll_af)
                            else:
                                La = torch.tensor(0.0, device=self.A_star.device)
                        except Exception:
                            La = torch.tensor(0.0, device=self.A_star.device)
                        L_gap = torch.tensor(0.0, device=self.A_star.device)

                    # 2) Dr 上的保持项 Lr = |NLL(A*) - NLL(AD)|
                    if (self.alpha > 0.0) and (len(self.retain_data) > 0) and (self.AD is not None):
                        ridx = (global_step + step) % max(len(self.retain_data), 1)
                        r_end = min(ridx + self.batch_size, len(self.retain_data))
                        r_batch = self.retain_data[ridx:r_end]
                        r_images = [x["image"] for x in r_batch]
                        r_texts = [x["text"] for x in r_batch]
                        r_targets = [x["target"] for x in r_batch]
                        out_r = self.A_star.forward(r_images, r_texts, r_targets)
                        nll_astar_r = out_r.loss
                        with torch.no_grad():
                            nll_ad_r = self.AD.compute_nll(r_images, r_texts, r_targets)
                        L_retain = torch.abs(nll_astar_r - nll_ad_r)
                    else:
                        L_retain = torch.tensor(0.0, device=self.A_star.device)

                    # 组合损失：La + αLr + β|La - G|
                    loss = La + self.alpha * L_retain + self.beta * L_gap

                # 反向与优化（梯度累积 + AMP 梯度缩放）
                loss_to_log = loss
                loss = loss / self.grad_accum_steps
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                accum += 1
                if accum % self.grad_accum_steps == 0:
                    if self.scaler is not None and self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()

                total_loss += float(loss_to_log.item())
                total_steps += 1
                global_step += 1

                if (step + 1) % self.log_interval == 0 or (step + 1) == steps_total:
                    dt = time.time() - t0
                    ips = (total_steps * self.batch_size) / max(dt, 1e-6)
                    print(f"[Fusion][train] step {step+1}/{steps_total} | La={float(La.item()):.4f} Lr={float(L_retain.item()):.4f} L_gap={float(L_gap.item()):.4f} loss={float(loss_to_log.item()):.4f} | {ips:.1f} samples/s")

            epoch_loss = total_loss / max(total_steps, 1)
            print(f"[Fusion] Epoch {epoch+1} avg loss: {epoch_loss:.6f}")

            # ===== 遗忘层参数日志（Epoch 后 + Delta）=====
            if hasattr(self.A_star, "unlearning_layer") and (self.A_star.unlearning_layer is not None):
                for name, p in self.A_star.unlearning_layer.named_parameters():
                    data = p.detach()
                    mean_val = float(data.mean().item())
                    std_val = float(data.std(unbiased=False).item())
                    delta_l2 = float('nan')
                    try:
                        prev = self._unl_prev_state.get(name, None) if hasattr(self, "_unl_prev_state") else None
                        if prev is not None:
                            delta = data.detach().cpu() - prev
                            delta_l2 = float(delta.norm(p=2).item())
                    except Exception:
                        pass
                    logging.info(f"====[UNL][epoch {epoch+1} post] {name} mean={mean_val:.6f} std={std_val:.6f} delta_l2={delta_l2:.6f}====")

            # 监控当前在 Df 上的对齐误差均值 |La - G|（仅在启用时）
            if gap_enabled and (self.sigma > 0.0):
                with torch.no_grad():
                    align_err = 0.0
                    cnt = 0
                    for images, texts, targets, step, steps_total in self._iter_batches(self.forget_data):
                        nll_astar = float(self.A_star.compute_nll(images, texts, targets).item())
                        if self.Af is not None:
                            nll_af = float(self.Af.compute_nll(images, texts, targets).item())
                        else:
                            if (self.af_nll_batches is None) or (step >= len(self.af_nll_batches)):
                                raise RuntimeError("Af NLL cache不可用或越界，请先运行 --mode precompute_af_nll 并确保batch_size/debug_limit一致。")
                            nll_af = float(self.af_nll_batches[step])
                        La_val = abs(nll_astar - nll_af)
                        align_err += abs(La_val - float(self.baseline_gap))
                        cnt += 1
                    align_err /= max(cnt, 1)
                print(f"[Fusion] Current alignment error on Df: {align_err:.6f}")

                if (target_threshold is not None) and (align_err <= target_threshold):
                    print(f"[Fusion] Early stop: alignment error ({align_err:.6f}) <= threshold ({target_threshold:.6f})")
                    break
            else:
                print("[Fusion] Skipping gap-based early stop (disabled).")

        print("[Fusion] Training finished.")

def process_images(images):
    """保持原始数据格式（PIL/numpy/torch 或其嵌套列表），不强制 requires_grad，以避免无意义的警告。"""
    return images

def process_texts(texts):
    """保持原始文本列表/字符串，交由模型内部的 processor 处理。"""
    return texts