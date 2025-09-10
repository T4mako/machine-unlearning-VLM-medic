from sklearn.linear_model import LogisticRegression
import numpy as np
import torch


def member_inference_attack(model, retain_data, forget_data):
    """成员推断攻击
    采集模型的中间特征作为攻击者输入，二分类区分 retain 与 forget。
    修复：确保 X 为二维 (n_samples, n_features)，避免 sklearn LogisticRegression 的维度错误。
    """
    # 1. 收集隐藏状态（将每个样本压平为一维向量）
    hidden_states_retain = []
    hidden_states_forget = []

    with torch.no_grad():
        for batch in retain_data[:100]:  # 采样
            h = model.get_fused_features(batch["image"], batch["text"])  # 形状可能是 [1, D] 或 [D]
            if isinstance(h, torch.Tensor):
                h = h.detach().cpu()
            # 压平到一维特征向量
            if hasattr(h, "dim") and h.dim() > 1:
                h = h.view(-1)
            hidden_states_retain.append(h.numpy() if hasattr(h, "numpy") else np.asarray(h))

        for batch in forget_data[:100]:
            h = model.get_fused_features(batch["image"], batch["text"])  # 形状可能是 [1, D] 或 [D]
            if isinstance(h, torch.Tensor):
                h = h.detach().cpu()
            if hasattr(h, "dim") and h.dim() > 1:
                h = h.view(-1)
            hidden_states_forget.append(h.numpy() if hasattr(h, "numpy") else np.asarray(h))

    # 2. 构建攻击者数据集（二维矩阵：n_samples x n_features）
    X = np.vstack(hidden_states_retain + hidden_states_forget)
    y = np.array([0] * len(hidden_states_retain) + [1] * len(hidden_states_forget))  # 0:retain, 1:forget

    # 3. 训练二分类攻击者模型
    attacker = LogisticRegression(max_iter=1000)
    attacker.fit(X, y)

    # 4. 评估
    acc = attacker.score(X, y)
    print(f"MIA Attack Accuracy: {acc:.4f} (Closer to 0.5 is better!)")
    return acc