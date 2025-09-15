import math
from typing import List, Dict, Optional
import torch

from model.model_wrapper import GenerativeQwenVLModel


def _avg_abs_gap(M1: GenerativeQwenVLModel, M2: GenerativeQwenVLModel, data: List[Dict]) -> float:
    total = 0.0
    cnt = 0
    for i in range(0, len(data), 4):  # small batch for speed
        batch = data[i:i+4]
        images = [x["image"] for x in batch]
        texts = [x["text"] for x in batch]
        targets = [x["target"] for x in batch]
        nll1 = float(M1.compute_nll(images, texts, targets).item())
        nll2 = float(M2.compute_nll(images, texts, targets).item())
        total += abs(nll1 - nll2)
        cnt += 1
    return total / max(cnt, 1)


def _avg_nll(M: GenerativeQwenVLModel, data: List[Dict]) -> float:
    total = 0.0
    cnt = 0
    for i in range(0, len(data), 4):
        batch = data[i:i+4]
        images = [x["image"] for x in batch]
        texts = [x["text"] for x in batch]
        targets = [x["target"] for x in batch]
        nll = float(M.compute_nll(images, texts, targets).item())
        total += nll
        cnt += 1
    return total / max(cnt, 1)


def evaluate(A_star: GenerativeQwenVLModel,
             retain_data: List[Dict], forget_data: List[Dict], dn_data: List[Dict], val_data: List[Dict],
             AD: Optional[GenerativeQwenVLModel] = None,
             Af: Optional[GenerativeQwenVLModel] = None,
             An: Optional[GenerativeQwenVLModel] = None):
    report = {}

    # 1) 知识差距：对齐程度
    if AD is not None and Af is not None and An is not None:
        gap_star_df = _avg_abs_gap(A_star, Af, forget_data)
        gap_base_dn = _avg_abs_gap(AD, An, dn_data)
        align_err = abs(gap_star_df - gap_base_dn)
        report["KGA_align_error"] = align_err
        report["gap_star_df"] = gap_star_df
        report["gap_base_dn"] = gap_base_dn

    # 2) 性能保持：与 AD 在 Dr/Val 上的一致性（NLL差）
    if AD is not None:
        retain_consistency = abs(_avg_nll(A_star, retain_data) - _avg_nll(AD, retain_data))
        val_consistency = abs(_avg_nll(A_star, val_data) - _avg_nll(AD, val_data))
        report["retain_consistency"] = retain_consistency
        report["val_consistency"] = val_consistency

    # 3) 生成样例（查看）
    try:
        sample = val_data[:2] if len(val_data) >= 2 else retain_data[:2]
        if sample:
            imgs = [x["image"] for x in sample]
            txts = [x["text"] for x in sample]
            gens = A_star.generate(imgs, txts, max_length=64, temperature=0.7)
            report["samples"] = [{"prompt": t, "gen": g, "target": sample[i]["target"]} for i, (t, g) in enumerate(zip(txts, gens))]
    except Exception as e:
        report["samples_error"] = str(e)

    return report