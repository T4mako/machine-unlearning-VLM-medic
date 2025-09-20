from data.load_PubMedVision import prepare_datasets
from model.model_wrapper import GenerativeQwenVLModel
from train.trainer import KGATrainer
from eval import kga_eval
from config import config
import torch
import logging
from utils.log_config import setup_logging
import argparse


def main():
    setup_logging()  # 初始化一次，之后全局生效
    logging.info("程序启动中...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'train_main', 'eval_teacher', 'eval_reference'], default='full', help='分批运行模式：full=原逻辑；train_main=仅训练主模型；eval_teacher=仅教师相关评估；eval_reference=仅参考模型相关评估。')
    parser.add_argument('--baseline_gap', type=float, default=None, help='可选：覆盖 baseline_gap（当分批运行无法同时加载AD/An时可用）。')
    args, _ = parser.parse_known_args()

    # 1. 数据：生成式image-text-to-text，并包含 Dn  外部集
    retain_data, forget_data, dn_data, val_data = prepare_datasets()
    logging.info(f"数据已划分 - Retain: {len(retain_data)}, Forget: {len(forget_data)}, Dn: {len(dn_data)}, Val: {len(val_data)}")

    # 2. 模型：以 AD 权重初始化 A*（若提供checkpoint）
    A_star = GenerativeQwenVLModel(model_name=config.model.model_name, use_fast=config.model.use_fast)
    logging.info(f"模型初始化: {config.model.model_name} on device {A_star.device}")
    if config.kga.ad_checkpoint:
        try:
            state = torch.load(config.kga.ad_checkpoint, map_location=A_star.device)
            A_star.load_state_dict(state)
            logging.info(f"[INFO] A* initialized from AD checkpoint: {config.kga.ad_checkpoint}")
        except Exception as e:
            logging.warning(f"[WARN] Failed to load AD checkpoint for A*: {e}")

    # 3. 训练/评估：根据 mode 分批构建教师/基线模型
    if args.mode == 'full':
        # 保持原有逻辑：全部加载
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
            load_AD=True,
            load_Af=True,
            load_An=True,
            baseline_gap_override=args.baseline_gap,
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

    elif args.mode == 'train_main':
        # 仅训练主模型：不加载教师/基线模型，必要时用 baseline_gap 覆盖
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
            load_AD=False,
            load_Af=False,
            load_An=False,
            baseline_gap_override=args.baseline_gap,
        )
        trainer.train(epochs=config.train.epochs)
        report = {}

    elif args.mode == 'eval_teacher':
        # 仅加载教师与主模型，进行与教师相关的评估或预估 gap（若需要）
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
            load_AD=True,
            load_Af=False,
            load_An=False,
            baseline_gap_override=args.baseline_gap,
        )
        # 可选：此模式下只跑 teacher 相关评估逻辑（此处简单返回空报告）
        report = {}

    elif args.mode == 'eval_reference':
        # 仅加载参考与主模型，进行参考模型相关评估
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
            load_AD=False,
            load_Af=True,
            load_An=True,
            baseline_gap_override=args.baseline_gap,
        )
        # 可选：此模式下可用于预计算 baseline_gap（AD/An 不全时需传入 baseline_gap）
        report = {}

    print("=" * 60)
    for k_name, v in (report or {}).items():
        print(f"{k_name}: {v}")


if __name__ == "__main__":
    main()