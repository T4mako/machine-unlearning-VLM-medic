import torch
import torch.nn.functional as F
from model.model_wrapper import QwenVLWithUnlearning


class EULTrainer:
    def __init__(self, model, retain_data, forget_data, val_data, lr=5e-4):
        self.model = model
        # 只优化遗忘层的参数
        self.optimizer = torch.optim.Adam(model.unlearning_layer.parameters(), lr=lr)
        self.retain_data = retain_data
        self.forget_data = forget_data
        self.val_data = val_data

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

    def train(self, epochs=5):
        for epoch in range(epochs):
            # 交替训练
            if epoch % 2 == 0:
                # 偶数轮：在保留集上训练 (Dr)
                self._train_on_retain()
            else:
                # 奇数轮：在遗忘集上训练 (Df)
                self._train_on_forget()

    def _train_on_retain(self):
        self.model.train()
        total_loss = 0
        for batch in self.retain_data:  # 需要实现DataLoader
            images, texts = batch["image"], batch["text"]
            answer = batch["answer"]

            # 获取“戴眼镜的医生”(F')的输出
            outputs_student = self.model(images, texts)
            logits_student = outputs_student.logits  # 假设可以获取logits

            # 获取“没戴眼镜的医生”(F)的输出 (教师模型)
            with torch.no_grad():
                outputs_teacher = self.model.model(images, texts)  # 直接调用冻结的model
                logits_teacher = outputs_teacher.logits

            # L_KL: 最小化KL散度 (靠近教师)
            kl_loss = self.compute_kl_loss(logits_student, logits_teacher)

            # L_TASK: 任务损失 (答对) —— 确保标签为LongTensor
            if not torch.is_tensor(answer):
                answer = torch.tensor([answer], dtype=torch.long, device=logits_student.device)
            else:
                if answer.dim() == 0:
                    answer = answer.to(logits_student.device).long().unsqueeze(0)
                else:
                    answer = answer.to(logits_student.device).long()
            task_loss = F.cross_entropy(logits_student, answer)

            # 总损失
            loss = kl_loss + self.lambda_task * task_loss

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        print(f"Retain Epoch Loss: {total_loss / len(self.retain_data)}")

    def _train_on_forget(self):
        self.model.train()
        total_loss = 0
        for batch in self.forget_data:
            images, texts = batch["image"], batch["text"]
            answer = batch["answer"]

            # 获取“戴眼镜的医生”(F')的输出
            outputs_student = self.model(images, texts)
            logits_student = outputs_student.logits

            # 获取“没戴眼镜的医生”(F)的输出 (教师模型)
            with torch.no_grad():
                outputs_teacher = self.model.model(images, texts)
                logits_teacher = outputs_teacher.logits

            # L_KL: 最大化KL散度 -> 最小化负KL散度
            kl_loss = -self.compute_kl_loss(logits_student, logits_teacher)

            # L_MMR: 破坏记忆 (这里需要修改输入为带[MASK]的问题)
            # 伪代码: texts_with_mask = add_mask(texts)
            # mmr_loss = ... # 计算预测[MASK]的损失

            # 假设mmr_loss已计算
            # loss = kl_loss + self.gamma_mmr * mmr_loss

            # 由于mmr_loss实现复杂，此处省略
            loss = kl_loss  # 简化

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        print(f"Forget Epoch Loss: {total_loss / len(self.forget_data)}")