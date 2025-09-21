from data.load_PubMedVision import prepare_datasets
from model.model_wrapper import GenerativeQwenVLModel
from train.trainer import KGATrainer
from eval import kga_eval
from config import config
import torch
import logging
from utils.log_config import setup_logging
import argparse
import os
import math


def _iter_batches(data, batch_size, debug_limit=None):
    dataset = data if debug_limit is None else data[: debug_limit]
    B = max(int(batch_size), 1)
    for i in range(0, len(dataset), B):
        batch_items = dataset[i: i + B]
        images = [x["image"] for x in batch_items]
        texts = [x["text"] for x in batch_items]
        targets = [x["target"] for x in batch_items]
        yield images, texts, targets


def compute_baseline_gap_singleton(model_name: str, ckpt_ad: str, ckpt_an: str, dn_data):
    """仅加载一次模型实例，顺序加载AD/An权重，按batch计算NLL并得到 baseline_gap。返回标量 G。"""
    model = GenerativeQwenVLModel(model_name=model_name, use_fast=config.model.use_fast)
    try:
        model.enable_unlearning(False)
    except Exception:
        pass

    def _load_ckpt(ckpt):
        if ckpt:
            state = torch.load(ckpt, map_location=model.device)
            model.load_state_dict(state)

    # 先跑 AD
    _load_ckpt(ckpt_ad)
    nll_ad = []
    for images, texts, targets in _iter_batches(dn_data, config.train.batch_size, getattr(config.train, 'debug_limit', None)):
        n = float(model.compute_nll(images, texts, targets).item())
        nll_ad.append(n)

    # 再跑 An
    _load_ckpt(ckpt_an)
    nll_an = []
    for images, texts, targets in _iter_batches(dn_data, config.train.batch_size, getattr(config.train, 'debug_limit', None)):
        n = float(model.compute_nll(images, texts, targets).item())
        nll_an.append(n)

    # E_batch[ |NLL_AD - NLL_An| ]
    total = 0.0
    cnt = max(len(nll_ad), 1)
    for a, b in zip(nll_ad, nll_an):
        total += abs(a - b)
    return total / cnt


def precompute_af_nll_singleton(model_name: str, ckpt_af: str, forget_data, out_path: str):
    """仅加载一次模型实例，加载Af权重，按batch计算在 Df 上的NLL均值并缓存到磁盘（逐样本nll同时保存便于校验）。"""
    model = GenerativeQwenVLModel(model_name=model_name, use_fast=config.model.use_fast)
    try:
        model.enable_unlearning(False)
    except Exception:
        pass
    if ckpt_af:
        state = torch.load(ckpt_af, map_location=model.device)
        model.load_state_dict(state)

    per_sample = []
    per_batch = []
    B = max(int(config.train.batch_size), 1)
    for i in range(0, len(forget_data), B):
        batch = forget_data[i:i + B]
        images = [x["image"] for x in batch]
        texts = [x["text"] for x in batch]
        targets = [x["target"] for x in batch]
        n = float(model.compute_nll(images, texts, targets).item())
        per_batch.append(n)
        # 逐样本近似：用batch均值填充（compute_nll当前返回批均loss）
        per_sample.extend([n] * len(batch))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "batch_nll": torch.tensor(per_batch, dtype=torch.float32),
        "nll": torch.tensor(per_sample, dtype=torch.float32),
        "meta": {
            "count_samples": len(per_sample),
            "count_batches": len(per_batch),
            "batch_size": B,
        }
    }
    torch.save(payload, out_path)
    logging.info(f"[KGA] Cached Af NLLs for Df -> {out_path} (batches={len(per_batch)}, samples={len(per_sample)})")


def main():
    setup_logging()  # 初始化一次，之后全局生效
    logging.info("程序启动中...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['full', 'train_main', 'eval_teacher', 'eval_reference', 'precompute_gap', 'train_with_af', 'precompute_af_nll'], default='full', help='分批运行模式：full=原逻辑；train_main=仅训练主模型；eval_teacher=仅教师相关评估；eval_reference=仅参考模型相关评估；precompute_gap=仅加载AD与An计算baseline_gap；train_with_af=仅加载A*并从缓存读取Af的NLL进行KGA训练；precompute_af_nll=仅加载Af预计算Df上的NLL缓存。')
    parser.add_argument('--baseline_gap', type=float, default=None, help='可选：覆盖 baseline_gap（当分批运行无法同时加载AD/An时可用）。')
    args, _ = parser.parse_known_args()

    # 1. 数据
    retain_data, forget_data, dn_data, val_data = prepare_datasets()
    logging.info(f"数据已划分 - Retain: {len(retain_data)}, Forget: {len(forget_data)}, Dn: {len(dn_data)}, Val: {len(val_data)}")

    # 分批模式分支
    if args.mode == 'precompute_gap':
        # 单实例顺序加载 AD/An，避免并发占用显存
        G = compute_baseline_gap_singleton(
            model_name=config.model.model_name,
            ckpt_ad=config.kga.ad_checkpoint,
            ckpt_an=config.kga.an_checkpoint,
            dn_data=dn_data,
        )
        os.makedirs('logs', exist_ok=True)
        gap_path = os.path.join('logs', 'baseline_gap.txt')
        with open(gap_path, 'w', encoding='utf-8') as f:
            f.write(str(G))
        logging.info(f"[KGA] baseline_gap saved to {gap_path}: {G:.6f}")
        report = {}

    elif args.mode == 'precompute_af_nll':
        # 单实例加载 Af，预计算 Df NLL 缓存
        out_path = os.path.join('logs', 'af_nll_forget.pt')
        precompute_af_nll_singleton(
            model_name=config.model.model_name,
            ckpt_af=config.kga.af_checkpoint,
            forget_data=forget_data,
            out_path=out_path,
        )
        report = {}

    elif args.mode == 'train_with_af':
        # 仅加载 A*，从缓存读取 Af 的 NLL 与预先保存的 baseline_gap 进行 KGA 训练
        # 允许命令行覆盖 baseline_gap；否则尝试从文件读取
        if args.baseline_gap is None:
            try:
                with open(os.path.join('logs', 'baseline_gap.txt'), 'r', encoding='utf-8') as f:
                    args.baseline_gap = float(f.read().strip())
                    logging.info(f"[KGA] Loaded baseline_gap from logs/baseline_gap.txt: {args.baseline_gap:.6f}")
            except Exception:
                logging.warning("[KGA] 未提供 --baseline_gap 且读取文件失败，将以0代替。建议先运行 --mode precompute_gap。")

        A_star = GenerativeQwenVLModel(model_name=config.model.model_name, use_fast=config.model.use_fast)
        logging.info(f"模型初始化: {config.model.model_name} on device {A_star.device}")
        if config.kga.ad_checkpoint:
            try:
                state = torch.load(config.kga.ad_checkpoint, map_location=A_star.device)
                A_star.load_state_dict(state)
                logging.info(f"[INFO] A* initialized from AD checkpoint: {config.kga.ad_checkpoint}")
            except Exception as e:
                logging.warning(f"[WARN] Failed to load AD checkpoint for A*: {e}")

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
            load_Af=False,  # 关键：不加载 Af，依赖缓存
            load_An=False,
            baseline_gap_override=args.baseline_gap,
            af_nll_path=os.path.join('logs', 'af_nll_forget.pt'),
        )
        trainer.train(epochs=config.train.epochs)
        report = {}

    else:
        # 其余模式保持原有逻辑（需要 A*）
        A_star = GenerativeQwenVLModel(model_name=config.model.model_name, use_fast=config.model.use_fast)
        logging.info(f"模型初始化: {config.model.model_name} on device {A_star.device}")
        if config.kga.ad_checkpoint:
            try:
                state = torch.load(config.kga.ad_checkpoint, map_location=A_star.device)
                A_star.load_state_dict(state)
                logging.info(f"[INFO] A* initialized from AD checkpoint: {config.kga.ad_checkpoint}")
            except Exception as e:
                logging.warning(f"[WARN] Failed to load AD checkpoint for A*: {e}")

        if args.mode == 'full':
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
            report = {}

        elif args.mode == 'eval_reference':
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
            report = {}

    print("=" * 60)
    for k_name, v in (report or {}).items():
        print(f"{k_name}: {v}")


if __name__ == "__main__":
    main()