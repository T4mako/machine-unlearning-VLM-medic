from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    use_fast: bool = False  # 是否使用 fast 版处理器
    temperature: float = 0.7  # 生成温度（分类任务暂不使用，保留做推理/生成时用）
    num_classes: Optional[int] = None  # 若为 None，则自动探测数据集类别数


@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 100
    lr: float = 5e-4
    log_interval: int = 5
    debug_limit: Optional[int] = 200  # 仅跑前 N 条样本做热身，确认流程


@dataclass
class EvalConfig:
    sample_size: int = 200  # 评估时使用的样本子集大小


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


# 全局配置对象
config = Config()