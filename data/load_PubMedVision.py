from typing import List, Dict, Tuple, Optional, Any
import random
import os
import pickle
import json
from io import BytesIO
from PIL import Image as PILImage
from config import config
import logging
from datasets import load_dataset
try:
    from datasets import load_from_disk
except Exception:
    load_from_disk = None
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None


_CACHE_DIR = os.path.join("logs", "cache")
_INTERMEDIATE_DIR = os.path.join("logs", "intermediate")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.makedirs(_INTERMEDIATE_DIR, exist_ok=True)

_DATASET_LOCAL_ROOT = None


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
    if hf_hub_download is not None:
        try:
            abs_path = hf_hub_download(
                repo_id="FreedomIntelligence/PubMedVision",
                filename=rel_path,
                repo_type="dataset",
            )
            return abs_path
        except Exception:
            return ""
    return ""


def _ensure_pil(img_obj):
    if isinstance(img_obj, PILImage.Image):
        return img_obj
    if isinstance(img_obj, dict):
        if "bytes" in img_obj and isinstance(img_obj["bytes"], (bytes, bytearray)):
            try:
                return PILImage.open(BytesIO(img_obj["bytes"]))
            except Exception:
                return None
        if "path" in img_obj:
            p = img_obj["path"]
            rp = _resolve_repo_file_path(p, only_local=True)
            if rp:
                try:
                    return PILImage.open(rp).convert("RGB")
                except Exception:
                    return None
    if isinstance(img_obj, str):
        rp = _resolve_repo_file_path(img_obj, only_local=True)
        if rp:
            try:
                return PILImage.open(rp).convert("RGB")
            except Exception:
                return None
    return None


def _serialize_image(img_obj) -> Any:
    """保留原始图像字段，不转 PIL"""
    return img_obj


def _deserialize_image(img_obj) -> Optional[PILImage.Image]:
    return _ensure_pil(img_obj)


def _cache_path(dn_ratio: float, debug_limit: int) -> str:
    key = f"pubmedvision_vqa_dn{dn_ratio}_dbg{debug_limit}.pkl"
    return os.path.join(_CACHE_DIR, key)


def _intermediate_path(dn_ratio: float, debug_limit: int) -> str:
    key = f"intermediate_dn{dn_ratio}_dbg{debug_limit}.jsonl"
    return os.path.join(_INTERMEDIATE_DIR, key)


def _progress_path(dn_ratio: float, debug_limit: int) -> str:
    key = f"progress_dn{dn_ratio}_dbg{debug_limit}.json"
    return os.path.join(_INTERMEDIATE_DIR, key)


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


def _save_splits_to_cache(retain, forget, dn, val, dn_ratio: float, debug_limit: int):
    path = _cache_path(dn_ratio, debug_limit)
    try:
        payload = {
            "meta": {"dn_ratio": dn_ratio, "debug_limit": debug_limit},
            "splits": {
                "retain": retain,
                "forget": forget,
                "dn": dn,
                "val": val,
            }
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        print(f"[CACHE] Saved prepared splits to {path}")
    except Exception as e:
        logging.warning(f"Failed to save cache: {e}")


def _process_sample(ex: Dict) -> Optional[Dict]:
    modality = ex.get("modality")
    if modality is None or (isinstance(modality, str) and modality.strip() == ""):
        return None

    images = ex.get("image")
    if isinstance(images, (str, dict)):
        images = [images]
    if not images:
        return None

    # 检查至少有一个图像能转 PIL（但不实际转）
    has_valid = False
    for img in images:
        if _ensure_pil(img) is not None:
            has_valid = True
            break
    if not has_valid:
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
        if human_text is not None and gpt_text is not None:
            break
    if human_text is None or gpt_text is None:
        return None

    return {
        "image": [_serialize_image(img) for img in images],  # 保留原始格式
        "modality": str(modality).strip(),
        "human": human_text,
        "gpt": gpt_text,
    }


def _load_intermediate_samples(dn_ratio: float, debug_limit: int) -> List[Dict]:
    intermediate_file = _intermediate_path(dn_ratio, debug_limit)
    samples = []
    if not os.path.exists(intermediate_file):
        return samples
    with open(intermediate_file, "r") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def prepare_datasets() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    logging.info("prepare_datasets")
    if load_from_disk is None:
        raise ImportError("Please install 'datasets'")

    dn_ratio = getattr(getattr(config, "kga", object()), "dn_ratio", 0.1)
    debug_limit = int(getattr(getattr(config, "train", object()), "debug_limit", 0) or 0)
    if debug_limit <= 0:
        debug_limit = None  # 表示无限制
    logging.info(f"开始准备数据 | dn_ratio={dn_ratio} debug_limit={debug_limit}")

    # 1. 尝试从最终缓存加载
    cached = _load_splits_from_cache(dn_ratio, debug_limit or 0)
    if cached is not None:
        logging.info("使用缓存数据")
        return cached

    # 2. 设置数据集路径
    global _DATASET_LOCAL_ROOT
    _DATASET_LOCAL_ROOT = os.path.abspath("./data/PubMedVision_repo")
    logging.info(f"强制设置 _DATASET_LOCAL_ROOT: {_DATASET_LOCAL_ROOT}")

    try:
        ds = load_from_disk("./data/PubMedVision_Alignment_VQA")
    except Exception:
        logging.info("Fallback to load_dataset from Hub...")
        ds = load_dataset("FreedomIntelligence/PubMedVision", "PubMedVision_Alignment_VQA")
        try:
            ds.save_to_disk('./data/PubMedVision_Alignment_VQA')
        except Exception as e:
            logging.warning(f"save_to_disk failed: {e}")
        if snapshot_download is not None:
            try:
                _DATASET_LOCAL_ROOT = snapshot_download(
                    repo_id="FreedomIntelligence/PubMedVision",
                    repo_type="dataset",
                    local_dir="./data/PubMedVision_repo",
                    local_dir_use_symlinks=False,
                )
            except Exception as e:
                logging.warning(f"snapshot_download failed: {e}")
        if not _DATASET_LOCAL_ROOT:
            for base in ["./data/PubMedVision_repo", "./data/PubMedVision_Alignment_VQA", "./"]:
                base_abs = os.path.abspath(base)
                if os.path.exists(os.path.join(base_abs, "images")):
                    _DATASET_LOCAL_ROOT = base_abs
                    break

    split_name = "train" if "train" in ds else list(ds.keys())[0]
    dset = ds[split_name]
    total_samples = len(dset)
    logging.info(f"数据集总长度: {total_samples}")

    # 3. 加载已有中间样本和进度
    intermediate_file = _intermediate_path(dn_ratio, debug_limit or 0)
    progress_file = _progress_path(dn_ratio, debug_limit or 0)

    existing_samples = _load_intermediate_samples(dn_ratio, debug_limit or 0)
    last_index = 0
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                last_index = json.load(f)["last_index"]
        except:
            pass

    start_index = last_index
    if start_index < len(existing_samples):
        # 可能进度文件损坏，以实际样本数为准
        start_index = len(existing_samples)

    logging.info(f"从索引 {start_index} 开始处理，已有 {len(existing_samples)} 个有效样本")

    # 4. 流式处理并追加到 intermediate 文件
    skipped = {"modality": 0, "images": 0, "pil": 0, "conv": 0}
    processed_count = len(existing_samples)

    with open(intermediate_file, "a") as f_out:
        for i in range(start_index, total_samples):
            if debug_limit is not None and processed_count >= debug_limit:
                break

            ex = dset[i]
            sample = _process_sample(ex)
            if sample is not None:
                f_out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                f_out.flush()
                processed_count += 1
            else:
                # 简单统计（可选）
                pass

            # 保存进度
            if (i + 1) % 1000 == 0:
                with open(progress_file, "w") as f_prog:
                    json.dump({"last_index": i + 1}, f_prog)
                logging.info(f"Processed {i + 1}/{total_samples}, valid: {processed_count}")

            if debug_limit is not None and processed_count >= debug_limit:
                break

    # 5. 重新加载所有中间样本
    all_samples = _load_intermediate_samples(dn_ratio, debug_limit or 0)
    if not all_samples:
        raise RuntimeError("No valid samples")

    # 6. 构建 items（此时仍不转 PIL）
    modalities = sorted({r["modality"] for r in all_samples})
    modality2id = {m: i for i, m in enumerate(modalities)}
    items = []
    for r in all_samples:
        items.append({
            "image": r["image"],
            "text": r["human"],
            "target": r["gpt"],
            "label": modality2id[r["modality"]],
        })

    # 7. 划分数据集（只操作元数据，内存小）
    random.seed(42)
    random.shuffle(items)
    n_total = len(items)
    n_val = max(1, int(0.1 * n_total))
    val_data_meta = items[:n_val]
    train_pool_meta = items[n_val:]

    forget_label = 0
    retain_meta = [it for it in train_pool_meta if it["label"] != forget_label]
    forget_meta = [it for it in train_pool_meta if it["label"] == forget_label]

    n_dn = max(1, int(dn_ratio * len(retain_meta))) if retain_meta else 0
    random.seed(42)
    dn_meta = random.sample(retain_meta, n_dn) if n_dn > 0 else []

    # 8. 最后一步：将元数据转为含 PIL 的最终数据（这里会吃内存！）
    def _meta_to_pil_dataset(meta_list: List[Dict]) -> List[Dict]:
        result = []
        for item in meta_list:
            pil_images = []
            for img_raw in item["image"]:
                pil = _deserialize_image(img_raw)
                if pil is not None:
                    pil_images.append(pil)
            if pil_images:  # 只保留有图像的
                result.append({
                    "image": pil_images,
                    "text": item["text"],
                    "target": item["target"],
                    "label": item["label"],
                })
        return result

    retain_dataset = _meta_to_pil_dataset(retain_meta)
    forget_dataset = _meta_to_pil_dataset(forget_meta)
    dn_dataset = _meta_to_pil_dataset(dn_meta)
    val_data = _meta_to_pil_dataset(val_data_meta)

    # 9. 保存最终缓存
    _save_splits_to_cache(retain_dataset, forget_dataset, dn_dataset, val_data, dn_ratio, debug_limit or 0)

    logging.info(f"Final sizes -> Dr={len(retain_dataset)}, Df={len(forget_dataset)}, Dn={len(dn_dataset)}, Val={len(val_data)}")
    return retain_dataset, forget_dataset, dn_dataset, val_data