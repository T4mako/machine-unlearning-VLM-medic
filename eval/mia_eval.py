from sklearn.linear_model import LogisticRegression
import numpy as np
import torch


def member_inference_attack(model, retain_data, forget_data):
    """成员推断攻击"""
    # 1. 收集隐藏状态
    hidden_states_retain = []
    hidden_states_forget = []

    with torch.no_grad():
        for batch in retain_data[:100]:  # 采样
            h = model.get_fused_features(batch["image"], batch["text"])
            hidden_states_retain.append(h.cpu().numpy())

        for batch in forget_data[:100]:
            h = model.get_fused_features(batch["image"], batch["text"])
            hidden_states_forget.append(h.cpu().numpy())

    # 2. 构建攻击者数据集
    X = np.vstack([np.array(hidden_states_retain), np.array(hidden_states_forget)])
    y = np.array([0] * len(hidden_states_retain) + [1] * len(hidden_states_forget))  # 0:retain, 1:forget

    # 3. 训练二分类攻击者模型
    attacker = LogisticRegression()
    attacker.fit(X, y)

    # 4. 评估
    acc = attacker.score(X, y)
    print(f"MIA Attack Accuracy: {acc:.4f} (Closer to 0.5 is better!)")
    return acc