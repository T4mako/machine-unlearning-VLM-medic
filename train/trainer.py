import math
import time
import torch
import torch.nn.functional as F
from typing import List, Dict
from config import config
from model.model_wrapper import GenerativeQwenVLModel


class KGATrainer:
    def __init__(self, A_star: GenerativeQwenVLModel,
                 retain_data: List[Dict], forget_data: List[Dict], dn_data: List[Dict], val_data: List[Dict],
                 lr=5e-6, batch_size=2, log_interval=5, debug_limit=None):
        self.A_star = A_star  # 待遗忘模型，初始化自 AD 权重
        # 简化：全参优化（可改LoRA/部分层）
        self.optimizer = torch.optim.AdamW(self.A_star.parameters(), lr=lr)

        self.retain_data = retain_data  # Dr
        self.forget_data = forget_data  # Df
        self.dn_data = dn_data          # Dn
        self.val_data = val_data
        self.batch_size = int(batch_size)
        self.log_interval = int(log_interval)
        self.debug_limit = int(debug_limit) if debug_limit is not None else None

        # 从配置读取KGA超参
        self.alpha = float(config.kga.alpha)
        self.sigma = float(config.kga.sigma)
        self.use_nll_gap = bool(config.kga.use_nll_gap)

        # 构建 AD、Af、An
        self.AD = self._load_model(config.kga.ad_checkpoint)
        self.Af = self._load_model(config.kga.af_checkpoint)
        self.An = self._load_model(config.kga.an_checkpoint)

        # 预计算基线差距 G = KL[AD(z), An(z)] 在 Dn 上的均值（用NLL差近似）
        self.baseline_gap = self._estimate_baseline_gap(self.AD, self.An, self.dn_data)
        print(f"[KGA] Baseline gap (G) on Dn: {self.baseline_gap:.6f}")

    def _load_model(self, ckpt_path):
        # 若提供checkpoint可在此load; 否则构造同权重模型
        model = GenerativeQwenVLModel(model_name=config.model.model_name, use_fast=config.model.use_fast)
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
            yield images, texts, targets, i // B, math.ceil(len(dataset) / B)

    @torch.no_grad()
    def _estimate_baseline_gap(self, M1: GenerativeQwenVLModel, M2: GenerativeQwenVLModel, data: List[Dict]):
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
        # 初始知识差距 G（用 AD 与 Af 在 Df 上的差距）
        with torch.no_grad():
            init_gap = self._estimate_baseline_gap(self.AD, self.Af, self.forget_data)
        print(f"[KGA] Initial gap on Df between AD and Af: {init_gap:.6f}")

        target_threshold = self.sigma * max(self.baseline_gap, 1e-8)
        print(f"[KGA] Early-stop threshold: {target_threshold:.6f}")

        global_step = 0
        for epoch in range(epochs):
            print(f"[KGA] Epoch {epoch+1}/{epochs}")
            # 一个epoch内，联合优化 La 和 Lr
            total_loss = 0.0
            total_steps = 0
            t0 = time.time()

            # 遗忘目标 La：让 A* 与 Af 在 Df 的gap 接近 AD 与 An 在 Dn 的gap
            for images, texts, targets, step, steps_total in self._iter_batches(self.forget_data):
                self.A_star.train()
                self.optimizer.zero_grad()

                # 当前的 gap_* = | NLL_A*(Df) - NLL_Af(Df) |
                nll_astar = self.A_star.compute_nll(images, texts, targets)
                with torch.no_grad():
                    nll_af = self.Af.compute_nll(images, texts, targets)
                gap_star = torch.abs(nll_astar - nll_af)

                # 基线 gap_base = | NLL_AD(Dn) - NLL_An(Dn) | (用预估均值代替逐batch)
                gap_base = torch.tensor(self.baseline_gap, device=self.A_star.device, dtype=gap_star.dtype)

                # 对齐损失：La = |gap_* - gap_base|
                La = torch.abs(gap_star - gap_base)

                # 性能保持 Lr：在 Dr 上让 A* 接近 AD
                # 简化：抽样一个 retain batch 同步计算 Lr
                if len(self.retain_data) > 0:
                    ridx = (global_step + step) % max(len(self.retain_data), 1)
                    r_end = min(ridx + self.batch_size, len(self.retain_data))
                    r_batch = self.retain_data[ridx:r_end]
                    r_images = [x["image"] for x in r_batch]
                    r_texts = [x["text"] for x in r_batch]
                    r_targets = [x["target"] for x in r_batch]
                    nll_astar_r = self.A_star.compute_nll(r_images, r_texts, r_targets)
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
                    print(f"[KGA][train] step {step+1}/{steps_total} | La={float(La.item()):.4f} Lr={float(Lr.item()):.4f} loss={float(loss.item()):.4f} | {ips:.1f} samples/s")

            epoch_loss = total_loss / max(total_steps, 1)
            print(f"[KGA] Epoch {epoch+1} avg loss: {epoch_loss:.6f}")

            # 监控当前在 Df 上的gap_* 均值，用于早停
            with torch.no_grad():
                current_gap = 0.0
                cnt = 0
                for images, texts, targets, step, steps_total in self._iter_batches(self.forget_data):
                    nll_astar = float(self.A_star.compute_nll(images, texts, targets).item())
                    nll_af = float(self.Af.compute_nll(images, texts, targets).item())
                    current_gap += abs(nll_astar - nll_af)
                    cnt += 1
                current_gap /= max(cnt, 1)
            print(f"[KGA] Current gap_* on Df: {current_gap:.6f}")

            if current_gap <= target_threshold:
                print(f"[KGA] Early stop: gap_* ({current_gap:.6f}) <= threshold ({target_threshold:.6f})")
                break

        print("[KGA] Training finished.")