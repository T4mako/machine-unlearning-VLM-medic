import json
import os
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from io import BytesIO
from PIL import Image as PILImage
from config import config
from datasets import load_dataset, Dataset
try:
    from datasets import load_from_disk
except ImportError:
    load_from_disk = None

# 轻量缓存目录（用于中间状态和最终缓存）
_CACHE_DIR = Path("logs") / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 中间处理目录
_INTERMEDIATE_DIR = Path("logs") / "intermediate"
_INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)

_DATASET_LOCAL_ROOT: Optional[str] = None


def _resolve_repo_file_path(rel_path: str, only_local: bool = False) -> str:
    if not isinstance(rel_path, str) or not rel_path:
        return ""
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    if _DATASET_LOCAL_ROOT:
        join_try = os.path.normpath(os.path.join(_DATASET_LOCAL_ROOT, rel_path))
        if os.path.exists(join_try):
            return join_try
    local_try = os.path.abspath(rel_path)
    if os.path.exists(local_try):
        return local_try
    return ""


def _serialize_image(img_obj) -> Dict[str, Any]:
    """将图像对象转为可序列化的形式（路径或 bytes）"""
    if isinstance(img_obj, str):
        return {"type": "path", "value": img_obj}
    if isinstance(img_obj, dict):
        if "bytes" in img_obj and isinstance(img_obj["bytes"], (bytes, bytearray)):
            return {"type": "bytes", "value": list(img_obj["bytes"])}  # 转为 list 以便 JSON
        if "path" in img_obj:
            return {"type": "path", "value": img_obj["path"]}
    # 如果已经是 PIL，我们不缓存它，而是回退到原始字段（但通常不会到这里）
    return {"type": "unknown", "value": None}


def _deserialize_image(serialized_img: Dict[str, Any]) -> Optional[PILImage.Image]:
    """从序列化形式恢复 PIL 图像"""
    img_type = serialized_img.get("type")
    value = serialized_img.get("value")
    if img_type == "path":
        rp = _resolve_repo_file_path(value, only_local=True)
        if rp:
            try:
                return PILImage.open(rp).convert("RGB")
            except Exception:
                return None
    elif img_type == "bytes":
        try:
            byte_data = bytes(value)  # value 是 list of int
            return PILImage.open(BytesIO(byte_data)).convert("RGB")
        except Exception:
            return None
    return None


def _get_progress_file(dn_ratio: float, debug_limit: int) -> Path:
    return _INTERMEDIATE_DIR / f"progress_dn{dn_ratio}_dbg{debug_limit}.json"


def _get_intermediate_file(dn_ratio: float, debug_limit: int) -> Path:
    return _INTERMEDIATE_DIR / f"intermediate_dn{dn_ratio}_dbg{debug_limit}.jsonl"


def _load_progress(dn_ratio: float, debug_limit: int) -> Tuple[int, List[Dict]]:
    progress_file = _get_progress_file(dn_ratio, debug_limit)
    if not progress_file.exists():
        return 0, []
    try:
        with open(progress_file, "r") as f:
            data = json.load(f)
        return data["last_index"], data["valid_samples"]
    except Exception as e:
        logging.warning(f"Failed to load progress: {e}")
        return 0, []


def _save_progress(last_index: int, valid_samples: List[Dict], dn_ratio: float, debug_limit: int):
    progress_file = _get_progress_file(dn_ratio, debug_limit)
    with open(progress_file, "w") as f:
        json.dump({"last_index": last_index, "valid_samples": valid_samples}, f)


def _process_sample(ex: Dict, index: int) -> Optional[Dict]:
    modality = ex.get("modality")
    if modality is None or (isinstance(modality, str) and not modality.strip()):
        return None

    images = ex.get("image")
    if isinstance(images, (str, dict)):
        images = [images]
    if not images:
        return None

    # 序列化图像（不转 PIL！）
    serialized_images = []
    for img in images:
        ser = _serialize_image(img)
        if ser["type"] != "unknown":
            serialized_images.append(ser)

    if not serialized_images:
        return None

    conv = ex.get("conversations") or []
    human_text, gpt_text = None, None
    for turn in conv:
        role = (turn.get("from") or "").lower()
        val = turn.get("value")
        if not isinstance(val, str) or not val.strip():
            continue
        if human_text is None and role == "human":
            human_text = val.strip()
        elif gpt_text is None and role == "gpt":
            gpt_text = val.strip()
        if human_text and gpt_text:
            break

    if not human_text or not gpt_text:
        return None

    return {
        "image": serialized_images,
        "modality": str(modality).strip(),
        "human": human_text,
        "gpt": gpt_text,
    }


def prepare_datasets() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    logging.info("prepare_datasets")
    if load_from_disk is None:
        raise ImportError("Please install 'datasets'")

    dn_ratio = getattr(getattr(config, "kga", object()), "dn_ratio", 0.1)
    debug_limit = int(getattr(getattr(config, "train", object()), "debug_limit", 0) or 0)
    logging.info(f"开始准备数据 | dn_ratio={dn_ratio} debug_limit={debug_limit}")

    # 尝试从最终缓存加载（pickle，含 PIL）—— 仅当用户明确需要时
    cached = _load_splits_from_cache(dn_ratio, debug_limit)
    if cached is not None:
        logging.info("使用最终缓存数据")
        return cached

    global _DATASET_LOCAL_ROOT
    _DATASET_LOCAL_ROOT = os.path.abspath("./data/PubMedVision_repo")
    logging.info(f"强制设置 _DATASET_LOCAL_ROOT: {_DATASET_LOCAL_ROOT}")

    # 加载数据集
    try:
        ds = load_from_disk("./data/PubMedVision_Alignment_VQA")
    except Exception:
        logging.info("Fallback to load_dataset from Hub...")
        ds = load_dataset("FreedomIntelligence/PubMedVision", "PubMedVision_Alignment_VQA")
        try:
            ds.save_to_disk('./data/PubMedVision_Alignment_VQA')
        except Exception as e:
            logging.warning(f"Failed to save dataset to disk: {e}")

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    dset = ds[split_name]
    total_samples = len(dset)
    logging.info(f"数据集总长度: {total_samples}")

    # 检查是否已有中间进度
    last_index, valid_samples = _load_progress(dn_ratio, debug_limit)
    if last_index > 0:
        logging.info(f"从断点恢复，已处理 {last_index} 个样本")

    # 流式处理
    batch_size = 100  # 每处理 100 个样本保存一次进度
    for i in range(last_index, total_samples):
        if debug_limit > 0 and len(valid_samples) >= debug_limit:
            break

        ex = dset[i]
        processed = _process_sample(ex, i)
        if processed is not None:
            valid_samples.append(processed)

        # 定期保存进度
        if (i + 1) % batch_size == 0 or i == total_samples - 1:
            _save_progress(i + 1, valid_samples, dn_ratio, debug_limit)
            logging.info(f"已处理 {i + 1}/{total_samples}，有效样本 {len(valid_samples)}")

        if debug_limit > 0 and len(valid_samples) >= debug_limit:
            break

    if not valid_samples:
        raise RuntimeError("No valid samples loaded.")

    # 构建 items（此时不加载 PIL，只保留序列化图像）
    modalities = sorted({r["modality"] for r in valid_samples})
    modality2id = {m: i for i, m in enumerate(modalities)}
    items = []
    for r in valid_samples:
        items.append({
            "image": r["image"],  # serialized
            "text": r["human"],
            "target": r["gpt"],
            "label": modality2id[r["modality"]],
        })

    # 划分数据集
    random.seed(42)
    random.shuffle(items)
    n_total = len(items)
    n_val = max(1, int(0.1 * n_total))
    val_data = items[:n_val]
    train_pool = items[n_val:]

    forget_label = 0
    retain_dataset = [it for it in train_pool if it["label"] != forget_label]
    forget_dataset = [it for it in train_pool if it["label"] == forget_label]

    n_dn = max(1, int(dn_ratio * len(retain_dataset))) if retain_dataset else 0
    random.seed(42)
    dn_dataset = random.sample(retain_dataset, n_dn) if n_dn > 0 else []

    # 可选：保存最终缓存（含 PIL）—— 注意：这会吃内存！
    # 如果你不需要快速重载，可以注释掉下面这行
    _save_splits_to_cache_with_pil(retain_dataset, forget_dataset, dn_dataset, val_data, dn_ratio, debug_limit)

    logging.info(f"Final sizes -> Dr={len(retain_dataset)}, Df={len(forget_dataset)}, Dn={len(dn_dataset)}, Val={len(val_data)}")
    return retain_dataset, forget_dataset, dn_dataset, val_data


def _save_splits_to_cache_with_pil(retain, forget, dn, val, dn_ratio: float, debug_limit: int):
    """仅当你确实需要缓存 PIL 对象时才调用（否则跳过）"""
    def _to_pil_item(item):
        pil_images = []
        for ser_img in item["image"]:
            pil = _deserialize_image(ser_img)
            if pil:
                pil_images.append(pil)
        if not pil_images:
            return None
        return {
            "image": pil_images,
            "text": item["text"],
            "target": item["target"],
            "label": item["label"],
        }

    def _filter_and_convert(lst):
        res = []
        for it in lst:
            pil_it = _to_pil_item(it)
            if pil_it:
                res.append(pil_it)
        return res

    try:
        retain_pil = _filter_and_convert(retain)
        forget_pil = _filter_and_convert(forget)
        dn_pil = _filter_and_convert(dn)
        val_pil = _filter_and_convert(val)

        path = _cache_path(dn_ratio, debug_limit)
        payload = {
            "meta": {"dn_ratio": dn_ratio, "debug_limit": debug_limit},
            "splits": {
                "retain": retain_pil,
                "forget": forget_pil,
                "dn": dn_pil,
                "val": val_pil,
            }
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[CACHE] Saved PIL splits to {path}")
    except Exception as e:
        logging.warning(f"Failed to save PIL cache: {e}")


def _load_splits_from_cache(dn_ratio: float, debug_limit: int):
    path = _cache_path(dn_ratio, debug_limit)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
        meta = payload.get("meta", {})
        if float(meta.get("dn_ratio", -1)) != float(dn_ratio) or int(meta.get("debug_limit", -1)) != int(debug_limit):
            return None
        splits = payload.get("splits", {})
        retain = splits.get("retain")
        forget = splits.get("forget")
        dn = splits.get("dn")
        val = splits.get("val")
        if any(x is None for x in (retain, forget, dn, val)):
            return None
        print(f"[CACHE] Loaded prepared splits from {path}")
        return retain, forget, dn, val
    except Exception as e:
        logging.warning(f"Cache load failed: {e}")
        return None


def _cache_path(dn_ratio: float, debug_limit: int) -> str:
    key = f"pubmedvision_vqa_dn{dn_ratio}_dbg{debug_limit}.pkl"
    return str(_CACHE_DIR / key)