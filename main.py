from venv import logger
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
from tqdm import tqdm

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
    """仅加载一次模型实例，顺序加载AD/An权重，按batch计算在 Dn 上的“知识差距”基线。
    注：理论上差距应为KL散度 E[KL(P_AD || P_An)]；出于显存与存储的可控性，这里采用常见的NLL差近似 E_batch[ |NLL_AD - NLL_An| ]，与训练阶段保持一致。
    """
    model = GenerativeQwenVLModel(model_name=model_name, use_fast=config.model.use_fast)
    try:
        model.enable_unlearning(False)
    except Exception:
        pass

    def _load_ckpt(ckpt):
        logging.info(f"==========[GAP] 尝试加载checkpoint: {ckpt}")
        if ckpt:
            try:
                if os.path.isdir(str(ckpt)):
                    try:
                        from peft import PeftModel
                        model.model = PeftModel.from_pretrained(model.model, str(ckpt))
                        logging.info(f"[GAP] 已加载LoRA适配器: {ckpt}")
                    except Exception as e:
                        logging.warning(f"[GAP] LoRA适配器加载失败，回退到全量权重: {e}")
                        state = torch.load(ckpt, map_location=model.device)
                        model.load_state_dict(state)
                else:
                    state = torch.load(ckpt, map_location=model.device)
                    model.load_state_dict(state)
            except Exception as e:
                logging.warning(f"[GAP] 加载checkpoint失败，将使用预训练权重: {e}")

    # 先跑 AD（完整数据 D 上训练的原始模型）
    _load_ckpt(ckpt_ad)
    nll_ad = []
    for images, texts, targets in _iter_batches(dn_data, config.train.batch_size, getattr(config.train, 'debug_limit', None)):
        n = float(model.compute_nll(images, texts, targets).item())
        nll_ad.append(n)

    # 再跑 An（在外部小型数据集 Dn 上训练的辅助模型，Dn 与 D 互不相交但分布相近）
    _load_ckpt(ckpt_an)
    nll_an = []
    for images, texts, targets in _iter_batches(dn_data, config.train.batch_size, getattr(config.train, 'debug_limit', None)):
        n = float(model.compute_nll(images, texts, targets).item())
        nll_an.append(n)

    # 近似：E_batch[ |NLL_AD - NLL_An| ]
    total = 0.0
    cnt = max(len(nll_ad), 1)
    for a, b in zip(nll_ad, nll_an):
        total += abs(a - b)
    return total / cnt


def compute_baseline_gap_dual(ad_model_name: str, ckpt_ad: str, an_model_name: str, ckpt_an: str, dn_data):
    """当 AD 与 An 的 checkpoint 属于不同架构（例如 An 为文本-only 学生）时，分开加载与评估，避免 state_dict 结构不匹配。
    计算 dis(AD, An, Dn) ≈ mean_NLL(AD, Dn) - mean_NLL(An, Dn)
    """
    import torch
    from model.model_wrapper import GenerativeQwenVLModel
    import logging

    # 先评估 AD 在 Dn 上的 NLL
    AD = GenerativeQwenVLModel(model_name=ad_model_name, use_fast=config.model.use_fast)
    try:
        AD.enable_unlearning(False)
    except Exception:
        pass
    if ckpt_ad:
        try:
            if os.path.isdir(str(ckpt_ad)):
                logging.info(f"==========[GAP] 尝试加载checkpoint: {ckpt}")
                try:
                    from peft import PeftModel
                    AD.model = PeftModel.from_pretrained(AD.model, str(ckpt_ad))
                    logging.info(f"[GAP] 已加载 AD LoRA 适配器: {ckpt_ad}")
                except Exception as e2:
                    logging.warning(f"[GAP] AD LoRA 加载失败，回退到全量权重: {e2}")
                    state = torch.load(ckpt_ad, map_location=AD.device)
                    AD.load_state_dict(state)
            else:
                state = torch.load(ckpt_ad, map_location=AD.device)
                AD.load_state_dict(state)
            logging.info(f"[GAP] 已加载 AD checkpoint/适配器: {ckpt_ad}")
        except Exception as e:
            logging.warning(f"[GAP] 加载 AD checkpoint 失败，将使用预训练权重: {e}")
    nll_ad = []
    for images, texts, targets in _iter_batches(dn_data, config.train.batch_size, getattr(config.train, 'debug_limit', None)):
        nll = AD.compute_nll(images, texts, targets)
        nll_ad.append(float(nll))
    try:
        del AD
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 再评估 An 在 Dn 上的 NLL
    An = GenerativeQwenVLModel(model_name=an_model_name, use_fast=config.model.use_fast,lora_enabled=False)
    try:
        An.enable_unlearning(False)
    except Exception:
        pass

    logging.info(f"[GAP] 尝试加载An checkpoint: {ckpt_an}")
    # 自动修正权重路径：如果是.pt文件，优先尝试同名目录
    ckpt_an_path = str(ckpt_an)
    if ckpt_an_path.endswith('.pt'):
        lora_dir = ckpt_an_path[:-3]
        if os.path.isdir(lora_dir):
            logging.info(f"[GAP] 检测到同名LoRA目录，优先加载: {lora_dir}")
            ckpt_an_path = lora_dir
    if ckpt_an_path:
        try:
            if os.path.isdir(ckpt_an_path):
                logging.info(f"[GAP] 发现 An checkpoint 目录: {ckpt_an_path}")
                try:
                    from peft import PeftModel
                    expected_files = ['adapter_config.json', 'adapter_model.safetensors']
                    for f in expected_files:
                        if not os.path.exists(os.path.join(ckpt_an_path, f)):
                            raise FileNotFoundError(f"缺少必需文件: {f}")
                    logging.info(f"[GAP] 开始加载 An LoRA 适配器...")
                    An.model = PeftModel.from_pretrained(
                        An.model,
                        ckpt_an_path,
                        device_map={"": An.device} if hasattr(An, 'device') else None
                    )
                    logging.info(f"[GAP] 已加载 An LoRA 适配器: {ckpt_an_path}")
                except Exception as e2:
                    logging.error(f"[GAP] An LoRA 加载失败（{type(e2).__name__}）: {e2}")
                    logging.warning("[GAP] 尝试回退到全量权重加载...")
                    pt_file = ckpt_an_path + '.pt'
                    if os.path.isfile(pt_file):
                        state = torch.load(pt_file, map_location=An.device)
                        An.load_state_dict(state)
                        logging.info(f"[GAP] 已加载全量权重文件: {pt_file}")
                    else:
                        raise FileNotFoundError(f"未找到全量权重文件: {pt_file}")
            else:
                logging.info(f"[GAP] 尝试加载全量权重文件: {ckpt_an_path}")
                state = torch.load(ckpt_an_path, map_location=An.device)
                An.load_state_dict(state)
            logging.info(f"[GAP] 已加载 An checkpoint/适配器: {ckpt_an_path}")
        except Exception as e:
            logging.error(f"[GAP] 加载 An checkpoint 失败（{type(e).__name__}）: {e}")
            logging.warning("[GAP] 将使用预训练权重")
    nll_an = []
    for images, texts, targets in _iter_batches(dn_data, config.train.batch_size, getattr(config.train, 'debug_limit', None)):
        nll = An.compute_nll(images, texts, targets)
        nll_an.append(float(nll))
    try:
        del An
        torch.cuda.empty_cache()
    except Exception:
        pass

    # 计算均值差距
    import math
    gap = (sum(nll_ad) / max(len(nll_ad), 1)) - (sum(nll_an) / max(len(nll_an), 1))
    return float(gap)


def precompute_af_nll_singleton(model_name: str, ckpt_af: str, forget_data, out_path: str):
    """仅加载一次模型实例，加载Af权重（在 Df 上训练的辅助模型），按batch计算在 Df 上的NLL均值并缓存到磁盘（逐样本nll同时保存便于校验）。"""
    model = GenerativeQwenVLModel(model_name=model_name, use_fast=config.model.use_fast,lora_enabled=False)
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


# ================= 离线KD：准备伪标签 =================

def prepare_kd_labels(dataset, out_path: str, teacher_model_name: str, teacher_ckpt: str, max_len: int, temperature: float):
    """仅加载一次教师模型，为给定数据集生成伪标签并保存到磁盘。"""
    logging.info(f"[KD] {teacher_model_name} 模型准备伪标签 -> {out_path}")
    teacher = GenerativeQwenVLModel(model_name=teacher_model_name, use_fast=config.model.use_fast, load_in_4bit=False,lora_enabled=False)
    try:
        teacher.enable_unlearning(False)
    except Exception:
        pass
    if teacher_ckpt:
        try:
            state = torch.load(teacher_ckpt, map_location=teacher.device)
            teacher.load_state_dict(state)
            logging.info(f"[KD] 教师checkpoint已加载: {teacher_ckpt}")
        except Exception as e:
            logging.warning(f"[KD] 教师checkpoint加载失败，使用预训练权重: {e}")

    all_prompts = []
    all_labels = []
    batch_size = config.train.batch_size
    debug_limit = getattr(config.train, 'debug_limit', None)
    total_samples = len(dataset) if debug_limit is None else min(len(dataset), debug_limit)
    num_batches = (total_samples + batch_size - 1)
    for images, texts, _ in tqdm(
            _iter_batches(dataset, batch_size, debug_limit),
            total=num_batches,
            desc="[KD] 生成伪标签",
            dynamic_ncols=True
    ):
        assert images is not None and len(images) == len(texts), "[KD] images为空或数量不匹配，必须为多模态输入"
        gens = teacher.generate(images, texts, temperature=temperature)
        all_prompts.extend(texts)
        all_labels.extend(gens)

    # 保存
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        "prompts": all_prompts,
        "labels": all_labels,
        "meta": {"count": len(all_prompts), "batch_size": int(batch_size)}
    }, out_path)
    logging.info(f"[KD] 伪标签已保存: {out_path} | {len(all_labels)} 条")

    # 彻底释放教师模型与显存
    del teacher
    logging.info("[KD] 教师模型已释放，开始清理GPU显存...")
    try:
        torch.cuda.empty_cache()
        logging.info("[KD] GPU显存已清理完毕")
    except Exception:
        logging.warning("[KD] GPU显存清理失败")


def train_student_from_kd_labels(dataset, labels_path: str, out_ckpt: str, student_model_name: str = None, student_init_ckpt: str = None):
    """仅加载一次学生模型，使用磁盘伪标签进行监督训练并保存checkpoint。"""
    logging.info(f"[KD] 加载学生 模型: {student_model_name}")
    logging.info(f"[KD] {student_model_name} 模型加载 payload: {labels_path}")
    payload = torch.load(labels_path)
    logging.info(f"[KD] {student_model_name} 模型加载 prompt 和 label")
    kd_prompts = payload.get("prompts", [])
    kd_labels = payload.get("labels", [])
    logging.info(f"[KD] 示例 prompt: {kd_prompts[0] if kd_prompts else '无'}")
    logging.info(f"[KD] 示例 label: {kd_labels[0] if kd_labels else '无'}")
    assert len(kd_prompts) == len(kd_labels), "KD标签文件损坏：prompts/labels长度不一致"

    model_name_to_use = student_model_name
    student = GenerativeQwenVLModel(model_name=student_model_name, use_fast=config.model.use_fast,load_in_4bit=False,lora_enabled=True)
    logging.info(f"[KD] 学生模型已加载完毕: {model_name_to_use}")
    try:
        student.enable_unlearning(False)  # An/Af 不使用遗忘层
        logging.info(f"[KD] {student_model_name} 学生模型遗忘层已禁用")
    except Exception:
        pass
    if student_init_ckpt:
        try:
            logging.info(f"[KD] 尝试加载学生初始化checkpoint: {student_init_ckpt}")
            state = torch.load(student_init_ckpt, map_location=student.device)
            student.load_state_dict(state)
            logging.info(f"[KD] 学生初始化checkpoint已加载: {student_init_ckpt}")
        except Exception as e:
            logging.warning(f"[KD] 学生初始化checkpoint加载失败，使用预训练初始化: {e}")

    # 优化器：仅训练LoRA权重
    lora_params = [p for n, p in student.model.named_parameters() if p.requires_grad]
    if len(lora_params) == 0:
        logging.warning("[KD] 未发现可训练参数，回退到训练全部参数（可能未成功注入LoRA）")
        lora_params = list(student.parameters())
    optim = torch.optim.AdamW(lora_params, lr=config.train.lr)

    B = max(int(config.train.batch_size), 1)
    n_items = len(dataset)
    debug_limit = getattr(config.train, 'debug_limit', None)
    if debug_limit is not None:
        n_items = min(n_items, int(debug_limit))

    def _get_kd_target_slice(start, end):
        return kd_labels[start:end]

    metrics = []
    best_loss = float('inf')
    patience = getattr(config.train, 'early_stopping_patience', 10)
    no_improve_epochs = 0
    stop_epoch = None
    idx = 0
    for epoch in range(int(config.train.epochs)):
        total_loss = 0.0
        n_batches = 0
        idx = 0
        for i in range(0, n_items, B):
            batch = dataset[i: i + B]
            images = [x["image"] for x in batch]
            texts = [x["text"] for x in batch]
            targets = _get_kd_target_slice(i, min(i + B, n_items))
            assert images is not None and len(images) == len(texts), "[KD] images为空或数量不匹配，必须为多模态输入"
            loss = student.loss_on_batch(images, texts, targets)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            total_loss += float(loss.item())
            n_batches += 1
            if (n_batches % max(int(config.train.log_interval), 1)) == 0:
                logging.info(f"[KD][epoch {epoch}] step={n_batches} loss={total_loss / max(n_batches,1):.4f}")

        avg_loss = total_loss / max(n_batches,1)
        logging.info(f"[KD] epoch={epoch} avg_loss={avg_loss:.4f}")
        metrics.append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "steps": n_batches,
            "total_loss": total_loss
        })
        # 早停机制
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            logging.info(f"[KD] 早停计数: {no_improve_epochs}/{patience}")
            if no_improve_epochs >= patience:
                logging.info(f"[KD] 触发早停机制，提前终止训练于 epoch {epoch}")
                stop_epoch = epoch
                break

    # 保存训练指标到 logs/LoRA/
    import json, datetime
    log_dir = os.path.join("logs", "LoRA")
    os.makedirs(log_dir, exist_ok=True)
    # 生成文件名：时间+模型名+数据名+数据量
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = getattr(dataset, "name", "data")
    # 修复模型名中的非法字符
    safe_model_name = str(student_model_name).replace('/', '_').replace('\\', '_').replace(':', '_')
    log_file = f"{now_str}_{safe_model_name}_{dataset_name}_{n_items}.json"
    log_path = os.path.join(log_dir, log_file)
    log_data = {
        "metrics": metrics,
        "early_stopped": stop_epoch is not None,
        "stop_epoch": stop_epoch
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    logging.info(f"[KD] LoRA训练日志已保存: {log_path}")

    # 保存学生LoRA 适配器（目录或文件名）
    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    try:
        from peft import PeftModel
        if isinstance(student.model, PeftModel):
            # 若 out_ckpt 是 .pt 文件名，则改为目录（去除扩展名）
            save_dir = out_ckpt
            if os.path.splitext(save_dir)[1].lower() in ('.pt', '.bin'):
                save_dir = os.path.splitext(save_dir)[0]
            student.model.save_pretrained(save_dir)
            logging.info(f"[KD] LoRA 适配器已保存: {save_dir}")
        else:
            # 回退：保存全量权重（不推荐，但保持兼容）
            torch.save(student.state_dict(), out_ckpt)
            logging.info(f"[KD] 学生已保存(全量权重): {out_ckpt}")
    except Exception as e:
        # 无 peft 或保存失败则回退
        torch.save(student.state_dict(), out_ckpt)
        logging.warning(f"[KD] 保存LoRA失败，已保存全量权重: {out_ckpt} | err={e}")

    # 释放
    del student
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def main():
    setup_logging()  # 初始化一次，之后全局生效
    logging.info("程序启动中...")

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=[
        'distill_an_labels', # 仅生成An的KD伪标签
        'distill_an_train', # 仅用An伪标签训练学生
        'distill_af_labels', # 仅生成Af的KD伪标签
        'distill_af_train', # 仅用Af伪标签训练学生
        'precompute_gap', # 仅加载AD与An计算G
        'precompute_af_nll', # 仅加载Af预计算Df上的NLL缓存
        'train_with_af', # 仅加载A*并从缓存读取Af的NLL进行KGA训练
        'distill_an', 
        'distill_af',
    ], default='precompute_gap',
        help='分批运行模式：distill_an_labels=仅生成An的KD伪标签；distill_an_train=仅用An伪标签训练学生；distill_af_labels=仅生成Af的KD伪标签；distill_af_train=仅用Af伪标签训练学生；precompute_gap=仅加载AD与An计算G；precompute_af_nll=仅加载Af预计算Df上的NLL缓存；train_with_af=仅加载A*并从缓存读取Af的NLL进行KGA训练；distill_an/ distill_af 为一键流程（先标签后训练）。')
    parser.add_argument('--baseline_gap', type=float, default=None, help='可选：覆盖 baseline_gap（当分批运行无法同时加载AD/An时可用）。')
    args, _ = parser.parse_known_args()

    # 1. 数据
    retain_data, forget_data, dn_data, val_data = prepare_datasets()
    logging.info(f"数据已划分 - Retain: {len(retain_data)}, Forget: {len(forget_data)}, Dn: {len(dn_data)}, Val: {len(val_data)}")

    # 拆分后的KD模式：只生成伪标签
    if args.mode in ("distill_an_labels", "distill_af_labels"):
        role = 'an' if args.mode == 'distill_an_labels' else 'af'
        dataset = dn_data if role == 'an' else forget_data
        labels_path = os.path.join('logs', f'{role}_kd_labels.pt')
        teacher_model_name = config.kd.teacher_model_name
        teacher_ckpt = (config.kd.teacher_ckpt or config.kga.ad_checkpoint) # 基础权重
        prepare_kd_labels(
            dataset=dataset,
            out_path=labels_path,
            teacher_model_name=teacher_model_name,
            teacher_ckpt=teacher_ckpt,
            max_len=int(config.kd.gen_max_len),
            temperature=float(config.kd.gen_temperature)
        )
        logging.info(f"[KD] {role.upper()} 伪标签已生成 -> {labels_path}")
        student_name = config.kd.student_model_name
        train_student_from_kd_labels(
            dataset=dataset,
            labels_path=labels_path,
            out_ckpt=out_ckpt,
            student_model_name=student_name,
            student_init_ckpt=(config.kd.student_init_ckpt or None)
        )
        logging.info(f"[KD] {role.upper()} 蒸馏流程完成 -> {out_ckpt}")
        return

    # 拆分后的KD模式：只训练学生
    if args.mode in ("distill_an_train", "distill_af_train"):
        role = 'an' if args.mode == 'distill_an_train' else 'af'
        logging.info(f"[KD] {role.upper()} 开始训练学生模型")
        dataset = dn_data if role == 'an' else forget_data
        labels_path = os.path.join('logs', f'{role}_kd_labels.pt')
        out_ckpt = (config.kd.an_out_ckpt if role == 'an' else config.kd.af_out_ckpt)
        student_name = config.kd.student_model_name
        train_student_from_kd_labels(
            dataset=dataset,
            labels_path=labels_path,
            out_ckpt=out_ckpt,
            student_model_name=student_name,
            student_init_ckpt=(config.kd.student_init_ckpt or None)
        )
        logging.info(f"[KD] {role.upper()} 学生训练完成 -> {out_ckpt}")
        return

    # 新增KD模式
    if args.mode in ("distill_an", "distill_af"):
        role = 'an' if args.mode == 'distill_an' else 'af'
        dataset = dn_data if role == 'an' else forget_data
        labels_path = os.path.join('logs', f'{role}_kd_labels.pt')
        out_ckpt = (config.kd.an_out_ckpt if role == 'an' else config.kd.af_out_ckpt)
        teacher_model_name = (config.kd.teacher_model_name or config.model.model_name)
        teacher_ckpt = (config.kd.teacher_ckpt or config.kga.ad_checkpoint)
        prepare_kd_labels(
            dataset=dataset,
            out_path=labels_path,
            teacher_model_name=teacher_model_name,
            teacher_ckpt=teacher_ckpt,
            max_len=int(config.kd.gen_max_len),
            temperature=float(config.kd.gen_temperature)
        )
        logging.info(f"[KD] {role.upper()} 伪标签已生成 -> {labels_path}")
        student_name = config.kd.student_model_name
        train_student_from_kd_labels(
            dataset=dataset,
            labels_path=labels_path,
            out_ckpt=out_ckpt,
            student_model_name=student_name,
            student_init_ckpt=(config.kd.student_init_ckpt or None)
        )
        logging.info(f"[KD] {role.upper()} 蒸馏流程完成 -> {out_ckpt}")
        return

    # 分批模式分支
    if args.mode == 'precompute_gap':
        # 单实例顺序加载 AD/An，避免并发占用显存
        # 若未显式提供 An checkpoint，优先使用KD产物
        ckpt_an = config.kga.an_checkpoint
        if ckpt_an is None or not os.path.exists(str(ckpt_an)):
            candidate_file = config.kd.an_out_ckpt
            candidate_dir = os.path.splitext(candidate_file)[0]
            logging.info(f"[KGA] 未提供有效的An checkpoint，尝试使用KD产物{candidate_file}")
            if os.path.exists(candidate_file):
                logging.info(f"[KGA] 发现KD产物文件: {candidate_file}")
                ckpt_an = candidate_file
            elif os.path.exists(candidate_dir):
                ckpt_an = candidate_dir
                logging.info(f"[KGA] 发现KD产物目录: {ckpt_an}")
            if ckpt_an is not None:
                logging.info(f"[KGA] 使用KD产出的An: {ckpt_an}")
        # 根据 An 来源选择对应的模型名称（KD 产物通常为学生 LoRA 适配器）
        an_model_name = config.model.model_name
        try:
            if ckpt_an and (os.path.isdir(str(ckpt_an)) or os.path.abspath(ckpt_an) == os.path.abspath(config.kd.an_out_ckpt)):
                an_model_name = config.kd.student_model_name
        except Exception:
            pass
        gap = compute_baseline_gap_dual(
            ad_model_name=config.model.model_name,
            ckpt_ad=config.kga.ad_checkpoint,
            an_model_name=an_model_name,
            ckpt_an=ckpt_an,
            dn_data=dn_data,
        )
        logging.info(f"[KGA] 基线差距（Dn）: {gap:.6f}")
        # 将 gap 保存到磁盘，供后续使用（可选）
        os.makedirs('logs', exist_ok=True)
        torch.save({"baseline_gap": float(gap)}, os.path.join('logs', 'baseline_gap.pt'))
        return

    if args.mode == 'precompute_af_nll':
        # 若未显式提供 Af checkpoint，优先使用KD产物
        ckpt_af = config.kga.af_checkpoint
        if ckpt_af is None or not os.path.exists(str(ckpt_af)):
            candidate_file = config.kd.af_out_ckpt
            candidate_dir = os.path.splitext(candidate_file)[0]
            logging.info(f"[KGA] 未提供有效的Af checkpoint，尝试使用KD产物{candidate_file}")
            if os.path.exists(candidate_file):
                logging.info(f"[KGA] 发现KD产物文件: {candidate_file}")
                ckpt_af = candidate_file
            elif os.path.exists(candidate_dir):
                ckpt_af = candidate_dir
                logging.info(f"[KGA] 发现KD产物目录: {ckpt_af}")
            if ckpt_af is not None:
                logging.info(f"[KGA] 使用KD产出的Af: {ckpt_af}")
        # 选择 Af 对应的模型名称
        af_model_name = config.model.model_name
        try:
            if ckpt_af and os.path.abspath(ckpt_af) == os.path.abspath(config.kd.af_out_ckpt):
                af_model_name = config.kd.student_model_name or ('ibm-granite/granite-docling-258M')
        except Exception:
            pass
        out_path = os.path.join('logs', 'af_nll_forget.pt')
        precompute_af_nll_singleton(
            model_name=config.model.model_name,
            ckpt_af=config.kga.ad_checkpoint,
            forget_data=forget_data,
            out_path=out_path)
        return

    if args.mode == 'train_with_af':
        # 训练 A*，使用缓存的 Af NLL 与 baseline_gap（若未提供，则尝试从磁盘加载 precompute_gap 结果）
        baseline_gap_override = args.baseline_gap
        if baseline_gap_override is None:
            gap_file = os.path.join('logs', 'baseline_gap.pt')
            if os.path.exists(gap_file):
                try:
                    payload = torch.load(gap_file)
                    baseline_gap_override = float(payload.get('baseline_gap', None))
                    logging.info(f"[KGA] 从文件加载 baseline_gap: {baseline_gap_override}")
                except Exception as e:
                    logging.warning(f"[KGA] 读取 baseline_gap 文件失败: {e}")

        A_star = GenerativeQwenVLModel(model_name=config.model.model_name, use_fast=config.model.use_fast, enable_unl=True, load_in_4bit= False,lora_enabled=False)
        logging.info(f"【A*】模型初始化: {config.model.model_name} on device {A_star.device}")
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
            load_Af=True,  # 关键：不加载 Af，依赖缓存
            load_An=False,
            baseline_gap_override=baseline_gap_override,
            af_nll_path=os.path.join('logs', 'af_nll_forget.pt'),
        )
        trainer.train(epochs=config.train.epochs)
        report = {}

    print("=" * 60)
    for k_name, v in (report or {}).items():
        print(f"{k_name}: {v}")


if __name__ == "__main__":
    main()