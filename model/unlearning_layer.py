import torch
import torch.nn as nn
import logging

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
        mlp_out = self.mlp(x)
        residual_out = self.residual(x)
        out = mlp_out + residual_out
        # 打印遗忘层参数和输出
        logging.info(f"[DEBUG][UnlearningLayer] mlp.weight: {[p.data for p in self.mlp.parameters()]}")
        logging.info(f"[DEBUG][UnlearningLayer] residual.weight: {self.residual.weight.data}")
        logging.info(f"[DEBUG][UnlearningLayer] input: {x}")
        logging.info(f"[DEBUG][UnlearningLayer] mlp_out: {mlp_out}")
        logging.info(f"[DEBUG][UnlearningLayer] residual_out: {residual_out}")
        logging.info(f"[DEBUG][UnlearningLayer] output: {out}")
        return out