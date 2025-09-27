import logging
from math import log

import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import List, Union, Optional
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModelForImageTextToText
import torch.nn.functional as F
from config import config
from model.unlearning_layer import UnlearningLayer
from PIL import Image

# 新增：文本模型支持
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception:
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore

try:
    from torchvision.transforms.functional import to_pil_image
except Exception:
    to_pil_image = None

# 新增：QLoRA/PEFT 支持
try:
    from peft import LoraConfig, get_peft_model
except Exception:
    LoraConfig = None  # type: ignore
    get_peft_model = None  # type: ignore

try:
    import bitsandbytes as bnb  # noqa: F401
except Exception:
    bnb = None  # type: ignore

class GenerativeQwenVLModel(nn.Module):
    """
    1. generate: 生成
    2. compute_nll: 计算给定target的负对数似然（用于KGA知识差距计算）
    3. forward: 返回包含logits的对象以兼容旧接口
    4. loss_on_batch: 用于蒸馏训练（带梯度）。
    """

    def __init__(
        self,
        model_name: str,
        use_fast: bool = config.model.use_fast,
        # 可选的实例级低显存/LoRA/遗忘层配置（若为None则回退到全局config.model）
        precision: Optional[str] = None, # 精度，默认全局config.model.precision
        load_in_4bit: Optional[bool] = None, # 是否加载4bit模型
        gradient_checkpointing: Optional[bool] = None, # 是否开启梯度检查点
        lora_enabled: Optional[bool] = None, # 是否开启LoRA
        lora_r: Optional[int] = None, # LoRA的秩，默认全局config.model.lora_r
        lora_alpha: Optional[int] = None, # LoRA的缩放系数，默认全局config.model.lora_alpha
        lora_dropout: Optional[float] = None, # LoRA的dropout率，默认全局config.model.lora_dropout
        lora_target_modules: Optional[List[str]] = None, # LoRA的目标模块，默认全局config.model.lora_target_modules
        enable_unl: Optional[bool] = None, # 是否开启遗忘层，默认全局config.model.enable_unl
        unl_hidden_dim: Optional[int] = None, # 遗忘层的隐藏层维度，默认全局config.model.unl_hidden_dim
        max_image_res: Optional[int] = None, # 最大图片分辨率，默认全局config.model.max_image_res
    ):
        super().__init__()
        # 设备选择：优先使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 将配置固化到实例（None 则取全局默认）
        self.precision = (precision or config.model.precision)
        self.load_in_4bit = bool(config.model.load_in_4bit if load_in_4bit is None else load_in_4bit)
        self.gradient_checkpointing = bool(config.model.gradient_checkpointing if gradient_checkpointing is None else gradient_checkpointing)
        self.lora_enabled = bool(config.model.lora_enabled if lora_enabled is None else lora_enabled)
        self.lora_r = int(config.model.lora_r if lora_r is None else lora_r)
        self.lora_alpha = int(config.model.lora_alpha if lora_alpha is None else lora_alpha)
        self.lora_dropout = float(config.model.lora_dropout if lora_dropout is None else lora_dropout)
        self.lora_target_modules = list(config.model.lora_target_modules if lora_target_modules is None else lora_target_modules)
        self.enable_unl_cfg = bool(config.model.enable_unl if enable_unl is None else enable_unl)
        self.unl_hidden_dim_cfg = int(config.model.unl_hidden_dim if unl_hidden_dim is None else unl_hidden_dim)
        self.max_image_res = int(config.model.max_image_res if max_image_res is None else max_image_res)
        self.max_seq_len = config.model.max_seq_len  # 针对对话式长文本设置上下文上限

        self.text_only: bool = False
        # 按实例配置选择精度
        dtype = torch.bfloat16 if str(self.precision).lower() == "bf16" else torch.float16

        # 优先加载多模态生成模型
        self.model = None
        self.processor = None
        # 4-bit 量化与设备映射配置（device_map/offload_folder 仍使用全局）
        device_map = getattr(config.model, "device_map", "auto")
        offload_folder = getattr(config.model, "offload_folder", "offload")


        try:
            self.model.config.use_cache = False
        except Exception:
            pass

        common_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "device_map": device_map,
            "offload_folder": offload_folder,
        }
        if self.load_in_4bit and (bnb is not None):
            common_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": dtype,
            })

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                **common_kwargs,
            )
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                use_fast=use_fast
            )
            logging.info("模型已加载（多模态 ImageTextToText）")
        except Exception as e:
            # 兼容：若 ImageTextToText 失败，尝试 Vision2Seq
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    **common_kwargs,
                )
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_fast=use_fast,
                )
                logging.info("模型已加载（多模态 Vision2Seq）")
            except Exception as e2:
                # 回退：文本-only 模型
                if AutoModelForCausalLM is None or AutoTokenizer is None:
                    raise
                logging.info(f"多模态加载失败，回退为文本-only模型: {e} | Vision2Seq失败: {e2}")
                self.text_only = True
                # 文本-only 也支持 4-bit
                text_kwargs = dict(common_kwargs)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **text_kwargs,
                )
                # 文本-only 使用 tokenizer 作为processor占位
                self.processor = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=use_fast,
                    trust_remote_code=True,
                )
                logging.info("模型已加载（文本-only CausalLM）")

        # ===== 遗忘层（两种模式：per-layer adapters 或 last_hidden） =====
        logging.info(f"模型{model_name}初始化遗忘层...")
        self.hidden_size = int(getattr(self.model.config, "hidden_size", 1024))
        self.unl_enabled: bool = bool(self.enable_unl_cfg)
        self.unl_hidden_dim: int = int(self.unl_hidden_dim_cfg)
        # 旧模式：仅在最后一层后接一个 UnlearningLayer
        self.unlearning_layer: Optional[UnlearningLayer] = None
        # 新模式：每层FFN后插入适配器
        self.unl_adapters: List[nn.Module] = []
        self.unl_mode: str = "disabled"
        if self.unl_enabled and self.hidden_size > 0:
            # 优先尝试 per-layer 适配器
            try:
                inserted = self._inject_unl_adapters()
                if inserted > 0:
                    self.unl_mode = "per_layer"
                    logging.info(f"[UNL] 已在Transformer各层FFN后注入适配器: {inserted} 个")
                else:
                    # 回退：仅使用最后一层的单适配器
                    self.unlearning_layer = UnlearningLayer(self.hidden_size, hidden_dim=self.unl_hidden_dim).to(self.device)
                    self.unl_mode = "last_hidden"
                    logging.info("[UNL] 未识别到可注入的FFN，回退为 last_hidden 模式")
            except Exception as e:
                logging.warning(f"[UNL] 注入per-layer适配器失败: {e}，回退为 last_hidden 模式")
                self.unlearning_layer = UnlearningLayer(self.hidden_size, hidden_dim=self.unl_hidden_dim).to(self.device)
                self.unl_mode = "last_hidden"

        # 冻结主干模型参数，只训练遗忘层/适配器
        if self.unl_enabled and (self.unl_mode != "disabled"):
            for param in self.model.parameters():
                param.requires_grad = False
            if self.unl_mode == "per_layer" and len(self.unl_adapters) > 0:
                for m in self.unl_adapters:
                    for p in m.parameters():
                        p.requires_grad = True
            elif self.unlearning_layer is not None:
                for param in self.unlearning_layer.parameters():
                    param.requires_grad = True

        # 禁用 use_cache 以兼容梯度检查点
        try:
            self.model.config.use_cache = False
        except Exception:
            pass

        # ===== 训练显存优化：LoRA & 梯度检查点 =====
        try:
            self.model.config.use_cache = False  # 训练禁用 KV cache
        except Exception:
            pass
        if self.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass

        # LoRA 仅在启用时注入，并且只训练 LoRA 权重（其余冻结）
        if self.lora_enabled and (LoraConfig is not None) and (get_peft_model is not None):
            logging.info(f"模型{model_name}初始化LoRA...")
            lora_cfg = LoraConfig(
                r=int(self.lora_r),
                lora_alpha=int(self.lora_alpha),
                lora_dropout=float(self.lora_dropout),
                target_modules=list(self.lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            for p in self.model.parameters():
                p.requires_grad = False
            # 仅 LoRA 权重训练
            for name, p in self.model.named_parameters():
                if "lora" in name:
                    p.requires_grad = True
            logging.info("已注入 LoRA 适配器并冻结主干参数")
        else:
            logging.info(f"模型{model_name}未启用LoRA")
    def enable_unlearning(self, enabled: bool = True):
        self.unl_enabled = bool(enabled)

    def get_unlearning_parameters(self):
        if hasattr(self, "unl_adapters") and len(self.unl_adapters) > 0:
            params: List[torch.nn.Parameter] = []
            for m in self.unl_adapters:
                params.extend(list(m.parameters()))
            return params
        return list(self.unlearning_layer.parameters()) if (self.unlearning_layer is not None) else []

    def _apply_unl(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """对最后一层隐藏状态应用遗忘层。last_hidden: [B, T, H]"""
        if (self.unl_enabled is False) or (self.unlearning_layer is None):
            return last_hidden
        B, T, H = last_hidden.shape
        x = last_hidden.reshape(-1, H)
        y = self.unlearning_layer(x)
        return y.reshape(B, T, H).clone()  # 确保输出连接到计算图

    # 新增：在各层FFN后注入遗忘适配器
    def _inject_unl_adapters(self) -> int:
        count = 0
        blocks = self._find_transformer_blocks()
        mlp_names = ["mlp", "feed_forward", "ffn", "ff", "mlp1", "mlp2"]
        for blk in blocks:
            for name in mlp_names:
                if hasattr(blk, name):
                    sub = getattr(blk, name)
                    if isinstance(sub, nn.Module):
                        adapter = UnlearningLayer(self.hidden_size, hidden_dim=self.unl_hidden_dim).to(self.device)
                        wrapped = nn.Sequential(sub, adapter)
                        setattr(blk, name, wrapped)
                        self.unl_adapters.append(adapter)
                        count += 1
                        break
        return count

    # 发现Transformer层（尽可能兼容不同结构）
    def _find_transformer_blocks(self) -> List[nn.Module]:
        blocks: List[nn.Module] = []
        m = self.model
        paths = [
            ("language_model", "model", "layers"),
            ("model", "layers"),
            ("transformer", "h"),
            ("transformer", "blocks"),
            ("decoder", "layers"),
        ]
        for path in paths:
            cur = m
            ok = True
            for attr in path:
                if hasattr(cur, attr):
                    cur = getattr(cur, attr)
                else:
                    ok = False
                    break
            if ok and isinstance(cur, (nn.ModuleList, list, tuple)) and len(cur) > 0:
                for b in cur:
                    blocks.append(b)
                if len(blocks) > 0:
                    return blocks
        # 回退：基于启发式扫描
        for mod in m.modules():
            if hasattr(mod, "mlp") or hasattr(mod, "feed_forward") or hasattr(mod, "ffn") or hasattr(mod, "ff"):
                blocks.append(mod)
        # 去重
        uniq = []
        seen = set()
        for b in blocks:
            if id(b) not in seen:
                uniq.append(b)
                seen.add(id(b))
        return uniq

    # ===== 图像预处理保持不变（文本-only 时会忽略） =====
    def _ensure_pil_list(self, x):
        # 兼容已有调用
        if isinstance(x, list):
            return x
        if to_pil_image and torch.is_tensor(x):
            return [to_pil_image(x)]
        return [x]

    # 文本-only 输入准备（供 _prepare_inputs 调用）
    def _prepare_inputs_text_only(self, texts, targets: Optional[List[str]] = None):
        """文本-only 输入准备：返回包含 input_ids/attention_mask 以及可选 labels 的字典。"""
        tokenizer = self.processor  # type: ignore
        if isinstance(texts, str):
            texts = [texts]
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        if targets is not None:
            if isinstance(targets, str):
                targets = [targets]
            lab = tokenizer(
                targets,
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            labels = lab["input_ids"].to(self.device)
            if "attention_mask" in lab:
                mask = lab["attention_mask"].to(self.device)
                labels = labels.masked_fill(mask == 0, -100)
            inputs["labels"] = labels
        return inputs

    def _prepare_inputs(self, images=None, texts: Union[str, List[str]] = None, targets: Optional[Union[str, List[str]]] = None):
        """统一的输入准备：同时支持多模态与文本-only。"""
        if self.text_only:
            return self._prepare_inputs_text_only(texts, targets)
        # 多模态路径
        if isinstance(texts, str):
            texts = [texts]
        images_list = self._ensure_pil_list(images) if images is not None else None
        # 使用 AutoProcessor 处理图像与文本
        proc_kwargs = {
            "return_tensors": "pt",
            "padding": True,
        }
        if images_list is not None:
            proc_kwargs["images"] = images_list
        if texts is not None:
            proc_kwargs["text"] = texts
        enc = self.processor(**proc_kwargs)
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        # 处理 labels（若提供targets）
        if targets is not None:
            if isinstance(targets, str):
                targets = [targets]
            # 优先使用 processor 的 tokenizer（若存在）
            tok = getattr(self.processor, "tokenizer", None)
            if tok is None:
                # 回退：文本-only tokenizer（极端情况）
                tok = self.processor  # type: ignore
            lab = tok(
                targets,
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
                return_tensors="pt",
            )
            labels = lab["input_ids"].to(self.device)
            if "attention_mask" in lab:
                mask = lab["attention_mask"].to(self.device)
                labels = labels.masked_fill(mask == 0, -100)
            inputs["labels"] = labels
        return inputs

    def compute_nll(self, images=None, texts: Union[str, List[str]] = None, targets: Union[str, List[str]] = None):
        """计算给定 batch 的平均 NLL。"""
        out = self.forward(images, texts, targets)
        if out.loss is None:
            raise RuntimeError("compute_nll 需要提供 targets 以计算loss")
        return out.loss

    def forward(self, images=None, texts: Union[str, List[str]] = None, targets: Union[str, List[str]] = None):
        """兼容旧训练接口：返回包含 loss 的对象（SimpleNamespace），以便 trainer 统一处理。"""
        
        if texts is None:
            raise ValueError("texts 不能为空")
        inputs = self._prepare_inputs(images, texts, targets)
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        # 获取最后层隐藏状态并应用遗忘层（如启用）
        last_hidden = out.hidden_states[-1]
        logits = None
        # 当采用 per-layer 适配器时，模型内部已被修改，此处直接使用 out.logits
        if hasattr(self, "unl_mode") and self.unl_mode == "per_layer":
            logits = out.logits
        elif self.unl_enabled and (self.unlearning_layer is not None) and (len(getattr(self, "unl_adapters", [])) == 0):
            last_hidden = self._apply_unl(last_hidden)
            head = None
            try:
                head = self.model.get_output_embeddings()
            except Exception:
                head = getattr(self.model, "lm_head", None)
                if head is None:
                    head = getattr(getattr(self.model, "language_model", None), "lm_head", None)
            if head is not None:
                logits = head(last_hidden)
            else:
                logits = out.logits
        else:
            logits = out.logits
        labels = inputs.get("labels")
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="mean")
        return SimpleNamespace(logits=logits, loss=loss)

    def loss_on_batch(self, images, texts: Union[str, List[str]], targets: Union[str, List[str]]):
        inputs = self._prepare_inputs(images, texts, targets)
        out = self.model(**inputs, output_hidden_states=True, use_cache=False)
        last_hidden = out.hidden_states[-1]
        # 当采用 per-layer 适配器时，模型内部已被修改，此处直接使用 out.logits
        if hasattr(self, "unl_mode") and self.unl_mode == "per_layer":
            logits = out.logits
        elif self.unl_enabled and (self.unlearning_layer is not None) and (len(getattr(self, "unl_adapters", [])) == 0):
            last_hidden = self._apply_unl(last_hidden)
            head = None
            try:
                head = self.model.get_output_embeddings()
            except Exception:
                head = getattr(self.model, "lm_head", None)
                if head is None:
                    head = getattr(getattr(self.model, "language_model", None), "lm_head", None)
            logits = head(last_hidden) if head is not None else out.logits
        else:
            logits = out.logits
        labels = inputs["labels"]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="mean")

    # 为兼容性保留的别名方法
    def get_fused_features(self, images, texts):
        """兼容旧接口：获取融合特征"""
        return self.get_hidden_states(images, texts)

    def forward_teacher(self, images, texts):
        """兼容旧接口：教师模型前向传播（等同于普通前向传播）"""
        return self.forward(images, texts)

    def get_hidden_states(self, images, texts):
        """获取模型隐藏状态，用于MIA等评估；文本-only 与多模态均返回最后一层平均池化。"""
        inputs = self._prepare_inputs(images, texts)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]
            pooled = last_hidden.mean(dim=1)  # [B, H]
        return pooled


import os
import logging
import torch
from peft import PeftModel

def auto_load_lora_or_pt(model, ckpt_path, device=None):
    """
    自动修正权重路径，优先加载LoRA适配器目录，回退.pt文件。
    model: 主干模型
    ckpt_path: 目录或.pt文件路径
    device: 加载到的设备
    返回加载后的模型
    """
    ckpt_path = str(ckpt_path)
    if ckpt_path.endswith('.pt'):
        lora_dir = ckpt_path[:-3]
        if os.path.isdir(lora_dir):
            logging.info(f"[权重加载] 检测到同名LoRA目录，优先加载: {lora_dir}")
            ckpt_path = lora_dir
    if os.path.isdir(ckpt_path):
        logging.info(f"[权重加载] 发现LoRA适配器目录: {ckpt_path}")
        expected_files = ['adapter_config.json', 'adapter_model.safetensors']
        for f in expected_files:
            if not os.path.exists(os.path.join(ckpt_path, f)):
                raise FileNotFoundError(f"缺少必需文件: {f}")
        model = PeftModel.from_pretrained(
            model,
            ckpt_path,
            device_map={"": device} if device else None
        )
        logging.info(f"[权重加载] 已加载LoRA适配器: {ckpt_path}")
        return model
    elif os.path.isfile(ckpt_path):
        logging.info(f"[权重加载] 尝试加载全量权重文件: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        logging.info(f"[权重加载] 已加载全量权重文件: {ckpt_path}")
        return model
    else:
        raise FileNotFoundError(f"未找到权重文件或目录: {ckpt_path}")