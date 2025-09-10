from data.load_medmnist import prepare_datasets
from model.model_wrapper import QwenVLWithUnlearning
from train.trainer import EULTrainer
from eval import utility_eval, forgetting_eval, mia_eval
from collections import Counter
from config import config


def main():
    # 1. 数据
    retain_data, forget_data, val_data = prepare_datasets()

    # 自动探测类别数与分布
    all_answers = [d["answer"] for d in retain_data] + [d["answer"] for d in forget_data] + [d["answer"] for d in val_data]
    label_set = sorted(set(all_answers))
    # 优先使用配置中的 num_classes，否则自动探测
    num_classes = config.model.num_classes if config.model.num_classes is not None else ((max(label_set) + 1) if label_set else 1)
    print(f"[INFO] Detected labels: {label_set}, num_classes={num_classes}")

    def print_dist(name, data):
        c = Counter(d["answer"] for d in data)
        c = dict(sorted(c.items()))
        print(f"[INFO] {name}: size={len(data)} label_dist={c}")

    print_dist("retain_data", retain_data)
    print_dist("forget_data", forget_data)
    print_dist("val_data", val_data)

    # 2. 模型（从配置读取参数）
    model = QwenVLWithUnlearning(model_name=config.model.model_name, num_classes=num_classes, use_fast=config.model.use_fast)

    # 3. 训练（从配置读取参数）
    trainer = EULTrainer(
        model,
        retain_data,
        forget_data,
        val_data,
        lr=config.train.lr,
        batch_size=config.train.batch_size,
        log_interval=config.train.log_interval,
        debug_limit=config.train.debug_limit,
    )
    trainer.train(epochs=config.train.epochs)

    # 4. 评估（采样子集大小从配置读取）
    k = config.eval.sample_size
    utility_eval.evaluate_utility(model, val_data[:k])
    forgetting_eval.evaluate_forgetting(model, forget_data[:k])
    mia_eval.member_inference_attack(model, retain_data[:k], forget_data[:k])


if __name__ == "__main__":
    main()