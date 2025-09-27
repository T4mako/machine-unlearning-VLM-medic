import torch
import torch.nn as nn
import logging

class UnlearningLayer(nn.Module):
    """轻量级的遗忘层，即“记忆过滤眼镜”"""

    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        # 一个简单的MLP（初始为近似零映射，以确保整体初始为恒等）
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)  # 输出维度与输入一致
        )
        # 使用残差连接，初始化为恒等映射
        self.residual = nn.Linear(input_dim, input_dim)
        nn.init.eye_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        # 将MLP初始化为零，使得初始状态 out ≈ x
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: 多模态融合后的表示 [*, input_dim]
        mlp_out = self.mlp(x)
        residual_out = self.residual(x)
        out = mlp_out + residual_out
        return out