from typing import List, Dict, Tuple
import random
import os
import pickle
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


# 轻量缓存目录（可存储PIL对象，使用pickle）
_CACHE_DIR = os.path.join("logs", "cache")
os.makedirs(_CACHE_DIR, exist_ok=True)

# 保存当前数据集在磁盘上的根目录，用于解析相对路径图片
_DATASET_LOCAL_ROOT = None

def _resolve_repo_file_path(rel_path: str) -> str:
    """尽量解析数据集相对路径到本地绝对路径。若失败返回空字符串。"""
    if not isinstance(rel_path, str) or not rel_path:
        return ""
    # 1) 绝对路径直接返回
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    # 2) 相对于数据集根目录解析（优先）
    if _DATASET_LOCAL_ROOT:
        join_try = os.path.normpath(os.path.join(_DATASET_LOCAL_ROOT, rel_path))
        if os.path.exists(join_try):
            return join_try
    # 3) 相对于当前工作目录解析
    local_try = os.path.abspath(rel_path)
    if os.path.exists(local_try):
        return local_try
    # 4) 回退到从 Hub 下载（仅当可用）
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
    """将输入转换为 PIL.Image.Image；若无法转换返回None。"""
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
            rp = _resolve_repo_file_path(p)
            if rp:
                try:
                    return PILImage.open(rp).convert("RGB")
                except Exception:
                    return None
    if isinstance(img_obj, str):
        rp = _resolve_repo_file_path(img_obj)
        if rp:
            try:
                return PILImage.open(rp).convert("RGB")
            except Exception:
                return None
    return None


def _cache_path(dn_ratio: float, debug_limit: int) -> str:
    key = f"pubmedvision_vqa_dn{dn_ratio}_dbg{debug_limit}.pkl"
    return os.path.join(_CACHE_DIR, key)


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
    except Exception:
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
    except Exception:
        pass


def prepare_datasets() -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """
    加载 FreedomIntelligence/PubMedVision（子集：PubMedVision_Alignment_VQA），
    """
    logging.info("prepare_datasets")
    if load_from_disk is None:
        raise ImportError("Please install 'datasets' to load FreedomIntelligence/PubMedVision from disk: pip install datasets")

    # 参数
    dn_ratio = getattr(getattr(config, "kga", object()), "dn_ratio", 0.1)
    debug_limit = int(getattr(getattr(config, "train", object()), "debug_limit", 0) or 0)

    logging.info(f"开始准备数据 | dn_ratio={dn_ratio} debug_limit={debug_limit}")

    # 优先从缓存加载
    cached = _load_splits_from_cache(dn_ratio, debug_limit)
    if cached is not None:
        logging.info("使用缓存数据")
        return cached

    # 1) 加载数据集（优先本地，失败则从 Hub 下载并保存本地）
    ds = None
    try:
        # 优先读取本地磁盘副本（满足你之前的需求）
        ds = load_from_disk("./data/PubMedVision_Alignment_VQA")
        logging.info("Loaded dataset from local disk: ./data/PubMedVision_Alignment_VQA")
    except Exception as e:
        logging.info(f"load_from_disk failed. Fallback to load_dataset from Hub...")
        ds = load_dataset("FreedomIntelligence/PubMedVision", "PubMedVision_Alignment_VQA")
        # 同步保存一份到本地，方便下次直接从磁盘读取
        try:
            ds.save_to_disk('./data/PubMedVision_Alignment_VQA')
        except Exception as se:
            print(f"[WARN][load_medmnist] save_to_disk failed: {se}")

    # 为后续相对路径图片解析设置本地根目录优先级
    global _DATASET_LOCAL_ROOT
    # 1) 若可用，下载整个数据集仓库快照（包含 images/ 子目录）
    if snapshot_download is not None:
        try:
            _DATASET_LOCAL_ROOT = snapshot_download(
                repo_id="FreedomIntelligence/PubMedVision",
                repo_type="dataset",
                local_dir="./data/PubMedVision_repo",
                local_dir_use_symlinks=False,
            )
            logging.info(f"Repo snapshot prepared at: {_DATASET_LOCAL_ROOT}")
        except Exception as re:
            logging.info(f"snapshot_download failed: {re}")
    # 2) 若仍未确定根目录，尝试启发式本地路径
    if not _DATASET_LOCAL_ROOT:
        for base in [
            "./data/PubMedVision_repo",
            "./data/PubMedVision_Alignment_VQA",
            "./",
        ]:
            base_abs = os.path.abspath(base)
            if os.path.exists(os.path.join(base_abs, "images")):
                _DATASET_LOCAL_ROOT = base_abs
                logging.info(f"Using heuristic dataset root: {_DATASET_LOCAL_ROOT}")
                break

    logging.info("加载数据...")
    split_name = "train" if "train" in ds else list(ds.keys())[0]
    dset = ds[split_name]
    logging.info("dset的长度: {}".format(len(dset)))

    # 2) 收集样本：改为保留该样本的所有图片(PIL 列表) + 第一对 human/gpt 文本
    records: List[Dict] = []
    total_seen = 0
    skipped_no_modality = 0
    skipped_no_images = 0
    skipped_no_pil_all = 0
    skipped_no_conv = 0
    for ex in dset:
        total_seen += 1
        logging.info("total_seen: {}".format(total_seen))
        modality = ex.get("modality")
        if modality is None or (isinstance(modality, str) and modality.strip() == ""):
            skipped_no_modality += 1
            continue
        images = ex.get("image")
        # 兼容单图场景：字符串/字典则转为列表
        if isinstance(images, (str, dict)):
            images = [images]
        if not images:
            logging.warning(f"NO.{total_seen} no images field or empty")
            skipped_no_images += 1
            continue
        # 将该样本的所有图片转为 PIL；若全部失败则跳过
        pil_images = []
        for img in images:
            pil_img = img if isinstance(img, PILImage.Image) else _ensure_pil(img)
            if pil_img is not None:
                pil_images.append(pil_img)
        if len(pil_images) == 0:
            logging.warning(f"NO.{total_seen} no pil images after resolution")
            skipped_no_pil_all += 1
            continue

        # 解析第一对 human/gpt
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
            skipped_no_conv += 1
            continue

        records.append({
            "image": pil_images,  # 注意：这里保存为列表，以支持原生多图
            "modality": str(modality).strip(),
            "human": human_text,
            "gpt": gpt_text,
        })
        logging.info("records length: {}".format(len(records)))

        # 调试加速：达到上限即停止收集
        if debug_limit > 0 and len(records) >= debug_limit:
            break

        # 周期性进度
        if total_seen % 500 == 0:
            print(f"[DEBUG][load_medmnist] scanned={total_seen} kept={len(records)} | no_modality={skipped_no_modality} no_images={skipped_no_images} no_pil={skipped_no_pil_all} no_conv={skipped_no_conv}")

    if len(records) == 0:
        raise RuntimeError("No valid samples with conversations loaded from PubMedVision.")

    avg_imgs = sum(len(r["image"]) for r in records) / max(len(records), 1)
    logging.info(f"Scan done: seen={total_seen}, kept={len(records)}, avg_imgs_per_sample={avg_imgs:.2f}")
    logging.info(f"Skipped summary | no_modality={skipped_no_modality}, no_images={skipped_no_images}, no_pil={skipped_no_pil_all}, no_conv={skipped_no_conv}")

    # 3) 模态到标签
    modalities = sorted({r["modality"] for r in records})
    modality2id = {m: i for i, m in enumerate(modalities)}
    # 统计每种模态样本数
    mod_counts = {}
    for r in records:
        m = r["modality"]
        mod_counts[m] = mod_counts.get(m, 0) + 1
    logging.info(f"Modalities discovered: {len(modalities)} -> {modalities}")
    logging.info(f"Samples per modality: {mod_counts}")

    # 4) 组装 items
    items: List[Dict] = []
    for r in records:
        lab = int(modality2id[r["modality"]])
        items.append({
            "image": r["image"],  # 此时为 List[PIL.Image]
            "text": r["human"],
            "target": r["gpt"],
            "label": lab,
        })

    # 5) 划分验证集（10%）
    random.seed(42)
    random.shuffle(items)
    n_total = len(items)
    n_val = max(1, int(0.1 * n_total))
    val_data = items[:n_val]
    train_pool = items[n_val:]
    logging.info(f"Total items={n_total} | val={len(val_data)} | train_pool={len(train_pool)}")

    # 6) 根据 label==0 切分 retain/forget，并构建 Dn
    forget_label = 0
    # 模态标签非0的样本为保留集
    retain_dataset = [it for it in train_pool if it["label"] != forget_label] 
    # 模态标签为0的样本为遗忘集
    forget_dataset = [it for it in train_pool if it["label"] == forget_label]
    logging.info(f"Split by forget_label={forget_label}: retain={len(retain_dataset)}, forget={len(forget_dataset)}")

    # Dn 子集（外部数据）
    n_dn = max(1, int(dn_ratio * len(retain_dataset))) if retain_dataset else 0
    random.seed(42)
    dn_dataset = random.sample(retain_dataset, n_dn) if n_dn > 0 else []
    logging.info(f"Built Dn subset: n_dn={len(dn_dataset)} (ratio={dn_ratio}) from retain={len(retain_dataset)}")

    # 保存到缓存（包含PIL对象）
    _save_splits_to_cache(retain_dataset, forget_dataset, dn_dataset, val_data, dn_ratio, debug_limit)

    logging.info(f"Final sizes -> Dr={len(retain_dataset)}, Df={len(forget_dataset)}, Dn={len(dn_dataset)}, Val={len(val_data)}")
    return retain_dataset, forget_dataset, dn_dataset, val_data