import os
import sys
# 保证无论从项目根目录还是在 eval/ 目录下直接运行，都能找到顶层包
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import math
import torch
from typing import List, Dict

from data.load_medmnist import prepare_datasets
from model.model_wrapper import GenerativeQwenVLModel as QwenVLWithUnlearning
from config import config


# 保留原接口签名，内部不会实际使用
def evaluate_dataset(model: QwenVLWithUnlearning, data: List[Dict], name: str, batch_size: int = 8) -> float:
    """
    使用老师模型（forward_teacher）对给定数据集进行预测并计算准确率。
    data: 形如 [{"image": ..., "text": ..., "answer": int}, ...]
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            images = [x["image"] for x in batch]
            texts = [x["text"] for x in batch]
            answers = [int(x["answer"]) for x in batch]

            outputs = model.forward_teacher(images, texts)
            logits = outputs.logits  # [B, C]
            preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()

            for p, t in zip(preds, answers):
                if int(p) == int(t):
                    correct += 1
                total += 1

    acc = correct / max(total, 1)
    print(f"Teacher Accuracy on {name}: {acc:.4f} (total={total})")
    return acc


def main():
    # 1) 准备数据（retain/forget/val）
    retain_data, forget_data, val_data = prepare_datasets()

    # 2) 自动探测类别数（若配置未显式指定）
    all_answers = [d["answer"] for d in retain_data] + [d["answer"] for d in forget_data] + [d["answer"] for d in val_data]
    label_set = sorted(set(all_answers))
    num_classes = config.model.num_classes if config.model.num_classes is not None else ((max(label_set) + 1) if label_set else 1)
    print(f"[INFO] Detected labels: {label_set}, num_classes={num_classes}")

    # 3) 构建老师模型（同主模型，但推理时走 forward_teacher）
    model = QwenVLWithUnlearning(
        model_name=config.model.model_name,
        num_classes=num_classes,
        use_fast=config.model.use_fast,
    )

    # 4) 分别评估三套数据
    bs = int(config.train.batch_size)
    acc_retain = evaluate_dataset(model, retain_data, name="retain", batch_size=bs)
    acc_forget = evaluate_dataset(model, forget_data, name="forget", batch_size=bs)
    acc_val = evaluate_dataset(model, val_data, name="val", batch_size=bs)

    # 5) 评估“所有数据”的整体准确率
    all_data = retain_data + forget_data + val_data
    acc_all = evaluate_dataset(model, all_data, name="all", batch_size=bs)

    print("=" * 60)
    print(f"Summary: retain={acc_retain:.4f}, forget={acc_forget:.4f}, val={acc_val:.4f}, all={acc_all:.4f}")


if __name__ == "__main__":
    main()