from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    model_name: str = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"
    model_base_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    # use_fast: bool = False  # 重复定义会导致 dataclass 报错；保留下方的 use_fast=True 作为默认
    temperature: float = 0.7  # 生成温度（分类任务暂不使用，保留做推理/生成时用）
    num_classes: Optional[int] = None  # 若为 None，则自动探测数据集类别数
    max_seq_len: int = 2048  # 最大序列长度
    use_fast: bool = True  # 是否使用 fast 版处理器
    # === 新增：遗忘层相关 ===
    enable_unl: bool = True           # 是否启用遗忘层（UnlearningLayer）
    unl_hidden_dim: int = 256          # 遗忘层瓶颈维度（MLP隐藏维）
    # === 新增：低显存训练开关 ===
    precision: str = "bf16"            # "bf16" 或 "fp16"
    load_in_4bit: bool = True          # 是否以 4-bit 量化加载（QLoRA）
    gradient_checkpointing: bool = True  # 是否开启梯度检查点
    device_map: str = "auto"           # 模型设备映射
    offload_folder: str = "offload"    # 当需要 CPU/NVMe offload 时的目录
    # LoRA 配置（仅当 lora_enabled=True 时生效）
    lora_enabled: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "o_proj"", k_proj", "gate_proj", "up_proj", "down_proj"
    ])
    # 图像与文本上限（用于进一步降低显存）
    max_image_res: int = 256           # 训练阶段统一缩放至不超过该分辨率的正方形


@dataclass
class TrainConfig:
    batch_size: int = 4
    epochs: int = 200
    lr: float = 5e-4
    log_interval: int = 5 
    debug_limit: Optional[int] = 1000  # 仅跑前 N 条样本做热身，确认流程
    # === 训练目标 & 冻结策略 ===
    objective: str = "fusion"         # 训练目标：融合（遗忘 + 保持 + 知识差距）
    freeze_backbone: bool = True      # 启用遗忘层时，是否冻结主干参数，仅训练遗忘层
    # === 低显存训练参数 ===
    gradient_accumulation_steps: int = 16  # 梯度累积步数
    use_8bit_optimizer: bool = False       # 是否使用 8-bit 优化器（bitsandbytes）
    early_stopping_patience: int = 20      # 早停轮数



@dataclass
class EvalConfig:
    sample_size: int = 200  # 评估时使用的样本子集大小


@dataclass
class KGAConfig:
    alpha: float = 1.0               # 保持项权重：L = L_forget + alpha * L_retain + beta * L_gap
    sigma: float = 0.2               # 早停阈值比例：对齐误差 <= sigma * |基线差距|
    dn_ratio: float = 0.1            # 从全体样本中划出外部集 Dn 比例（简化近似）
    ad_checkpoint: Optional[str] = None  # 原始模型 AD 的 checkpoint（如无则用基础模型权重）
    af_checkpoint: Optional[str] = None  # 辅助模型 Af（在 Df 上训练）
    an_checkpoint: Optional[str] = None  # 辅助模型 An（在 Dn 上训练）
    use_nll_gap: bool = True         # 用 NLL 差近似 KL 差
    # === 融合目标的权重 ===
    lambda_f: float = 1.0            # 遗忘项权重
    beta: float = 1.0                # 知识差距项权重


# 新增：知识蒸馏配置（离线KD）
@dataclass
class KDConfig:
    kd_type: str = "hard"                        # "hard"=伪标签KD；"soft"=分布蒸馏（需额外实现与更高显存）
    teacher_model_name: Optional[str] = "FreedomIntelligence/HuatuoGPT-Vision-7B-Qwen2.5VL"     # 若为None，默认使用 config.model.model_name
    student_model_name: Optional[str] = "Qwen/Qwen2-VL-2B-Instruct"     # 默认学生模型
    teacher_ckpt: Optional[str] = None           # 若为None，默认使用 config.kga.ad_checkpoint
    student_init_ckpt: Optional[str] = None      # 学生初始化权重（可选）
    an_out_ckpt: str = "weights/an_student.pt"      # 训练完成的 An 输出路径
    af_out_ckpt: str = "weights/af_student.pt"      # 训练完成的 Af 输出路径
    gen_max_len: int = 1024                       # 教师生成伪标签的最长长度
    gen_temperature: float = 0.7                 # 教师生成温度


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    kga: KGAConfig = field(default_factory=KGAConfig)
    kd: KDConfig = field(default_factory=KDConfig)


# 全局配置对象
config = Config()