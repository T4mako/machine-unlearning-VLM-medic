from data.load_medmnist import prepare_datasets
from model.model_wrapper import GenerativeQwenVLModel
from train.trainer import KGATrainer
from eval import kga_eval
from config import config
import torch
import logging
from utils.log_config import setup_logging

def main():
    setup_logging()  # 初始化一次，之后全局生效
    logger = logging.getLogger(__name__)
    logger.info("程序启动中...")

    # 1. 数据：生成式image-text-to-text，并包含 Dn  外部集
    retain_data, forget_data, dn_data, val_data = prepare_datasets()

    # 2. 模型：以 AD 权重初始化 A*（若提供checkpoint）
    A_star = GenerativeQwenVLModel(model_name=config.model.model_name, use_fast=config.model.use_fast)
    if config.kga.ad_checkpoint:
        try:
            state = torch.load(config.kga.ad_checkpoint, map_location=A_star.device)
            A_star.load_state_dict(state)
            print(f"[INFO] A* initialized from AD checkpoint: {config.kga.ad_checkpoint}")
        except Exception as e:
            print(f"[WARN] Failed to load AD checkpoint for A*: {e}")

    # 3. 训练：KGA
    trainer = KGATrainer(
        A_star=A_star,
        retain_data=retain_data,
        forget_data=forget_data,
        dn_data=dn_data,
        val_data=val_data,
        lr=config.train.lr,
        batch_size=config.train.batch_size,
        log_interval=config.train.log_interval,
        debug_limit=config.train.debug_limit,
    )
    trainer.train(epochs=config.train.epochs)

    # 4. 评估：KGA指标（知识差距对齐 + 性能保持）
    k = int(config.eval.sample_size)
    report = kga_eval.evaluate(
        A_star=A_star,
        retain_data=retain_data[:k],
        forget_data=forget_data[:k],
        dn_data=dn_data[:k],
        val_data=val_data[:k],
        AD=trainer.AD,
        Af=trainer.Af,
        An=trainer.An,
    )

    print("=" * 60)
    for k_name, v in report.items():
        print(f"{k_name}: {v}")


if __name__ == "__main__":
    main()