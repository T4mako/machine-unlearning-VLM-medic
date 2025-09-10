import math
import time
import torch
import torch.nn.functional as F
from model.model_wrapper import QwenVLWithUnlearning


class EULTrainer:
    def __init__(self, model, retain_data, forget_data, val_data, lr=5e-4, batch_size=8, log_interval=50, debug_limit=None):
        self.model = model
        # 只优化遗忘层的参数
        self.optimizer = torch.optim.Adam(model.unlearning_layer.parameters(), lr=lr)
        self.retain_data = retain_data
        self.forget_data = forget_data
        self.val_data = val_data
        self.batch_size = int(batch_size)
        self.log_interval = int(log_interval)
        self.debug_limit = int(debug_limit) if debug_limit is not None else None

        # 超参数
        self.alpha = 0.8
        self.lambda_task = 1.0
        self.gamma_mmr = 0.2

    def compute_kl_loss(self, student_logits, teacher_logits):
        """计算KL散度"""
        # 将logits转换为概率分布
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        # KL散度: KL(P||Q) = sum(P * log(P/Q))
        kl_loss = F.kl_div(F.log_softmax(student_logits, dim=-1), teacher_probs, reduction='batchmean')
        return kl_loss

    def _iter_batches(self, data):
        dataset = data if self.debug_limit is None else data[: self.debug_limit]
        B = self.batch_size
        for i in range(0, len(dataset), B):
            batch_items = dataset[i: i + B]
            images = [x["image"] for x in batch_items]
            texts = [x["text"] for x in batch_items]
            answers = torch.tensor([int(x["answer"]) for x in batch_items], dtype=torch.long, device=self.model.device)
            yield images, texts, answers, i // B, math.ceil(len(dataset) / B)

    def train(self, epochs=5):
        for epoch in range(epochs):
            print(f"[INFO] Epoch {epoch+1}/{epochs} - phase: {'retain' if epoch % 2 == 0 else 'forget'}")
            # 交替训练
            if epoch % 2 == 0:
                # 偶数轮：在保留集上训练 (Dr)
                self._train_on_retain()
            else:
                # 奇数轮：在遗忘集上训练 (Df)
                self._train_on_forget()

    def _train_on_retain(self):
        self.model.train()
        total_loss = 0.0
        count = 0
        t0 = time.time()
        for images, texts, answers, step, total_steps in self._iter_batches(self.retain_data):
            # 学生输出（带遗忘层）
            outputs_student = self.model(images, texts)
            logits_student = outputs_student.logits  # [B, C]

            # 教师输出（不带遗忘层）
            with torch.no_grad():
                outputs_teacher = self.model.forward_teacher(images, texts)
                logits_teacher = outputs_teacher.logits  # [B, C]

            # L_KL: 最小化KL散度 (靠近教师)
            kl_loss = self.compute_kl_loss(logits_student, logits_teacher)

            # L_TASK: 任务损失 (答对)
            task_loss = F.cross_entropy(logits_student, answers)

            # 总损失
            loss = kl_loss + self.lambda_task * task_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            count += 1

            if (step + 1) % self.log_interval == 0 or (step + 1) == total_steps:
                dt = time.time() - t0
                ips = (count * self.batch_size) / max(dt, 1e-6)
                print(f"[TRAIN][retain] step {step+1}/{total_steps} | loss={total_loss / count:.4f} | {ips:.1f} samples/s")
        print(f"Retain Epoch Loss: {total_loss / max(count,1):.6f}")

    def _train_on_forget(self):
        self.model.train()
        total_loss = 0.0
        count = 0
        t0 = time.time()
        for images, texts, answers, step, total_steps in self._iter_batches(self.forget_data):
            # 学生输出
            outputs_student = self.model(images, texts)
            logits_student = outputs_student.logits

            # 教师输出
            with torch.no_grad():
                outputs_teacher = self.model.forward_teacher(images, texts)
                logits_teacher = outputs_teacher.logits

            # L_KL: 最大化KL散度 -> 最小化负KL散度
            kl_loss = -self.compute_kl_loss(logits_student, logits_teacher)

            # 这里暂不加入 MMR，保持原简化
            loss = kl_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            count += 1

            if (step + 1) % self.log_interval == 0 or (step + 1) == total_steps:
                dt = time.time() - t0
                ips = (count * self.batch_size) / max(dt, 1e-6)
                print(f"[TRAIN][forget] step {step+1}/{total_steps} | loss={total_loss / count:.4f} | {ips:.1f} samples/s")
        print(f"Forget Epoch Loss: {total_loss / max(count,1):.6f}")