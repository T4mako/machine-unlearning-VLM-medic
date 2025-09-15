from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
    use_fast: bool = False  # 是否使用 fast 版处理器
    temperature: float = 0.7  # 生成温度（分类任务暂不使用，保留做推理/生成时用）
    num_classes: Optional[int] = None  # 若为 None，则自动探测数据集类别数


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 100
    lr: float = 5e-4
    log_interval: int = 5
    debug_limit: Optional[int] = 50  # 仅跑前 N 条样本做热身，确认流程


@dataclass
class EvalConfig:
    sample_size: int = 200  # 评估时使用的样本子集大小


@dataclass
class KGAConfig:
    alpha: float = 1.0               # L = La + alpha * Lr
    sigma: float = 0.2               # 早停阈值比例：对齐误差 <= sigma * |基线差距|
    dn_ratio: float = 0.1            # 从全体样本中划出外部集 Dn 比例（简化近似）
    ad_checkpoint: Optional[str] = None  # 原始模型 AD 的 checkpoint（如无则用基础模型权重）
    af_checkpoint: Optional[str] = None  # 辅助模型 Af（在 Df 上训练）
    an_checkpoint: Optional[str] = None  # 辅助模型 An（在 Dn 上训练）
    use_nll_gap: bool = True         # 用 NLL 差近似 KL 差


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    kga: KGAConfig = field(default_factory=KGAConfig)


# 全局配置对象
config = Config()