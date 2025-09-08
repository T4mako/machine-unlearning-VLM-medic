import torch
import torch.nn as nn


class UnlearningLayer(nn.Module):
    """轻量级的遗忘层，即“记忆过滤眼镜”"""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # 一个简单的MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # 输出维度与输入一致
        )
        # 使用残差连接，确保初始状态为恒等映射
        self.residual = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x: 多模态融合后的表示
        return self.mlp(x) + self.residual(x)