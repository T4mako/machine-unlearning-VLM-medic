import logging

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

        # ===== 遗忘层 =====
        logging.info(f"模型{model_name}初始化遗忘层...，但并未训练")
        self.hidden_size = int(getattr(self.model.config, "hidden_size", getattr(getattr(self.model, "config", object()), "hidden_size", 0)))
        self.unl_enabled: bool = bool(self.enable_unl_cfg)
        self.unl_hidden_dim: int = int(self.unl_hidden_dim_cfg)
        self.unlearning_layer: Optional[UnlearningLayer] = None
        if self.unl_enabled and self.hidden_size > 0:
            self.unlearning_layer = UnlearningLayer(self.hidden_size, hidden_dim=self.unl_hidden_dim).to(self.device)

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

    def enable_unlearning(self, enabled: bool = True):
        self.unl_enabled = bool(enabled)

    def get_unlearning_parameters(self):
        return list(self.unlearning_layer.parameters()) if (self.unlearning_layer is not None) else []

    def _apply_unl(self, last_hidden: torch.Tensor) -> torch.Tensor:
        """对最后一层隐藏状态应用遗忘层。last_hidden: [B, T, H]"""
        if (self.unl_enabled is False) or (self.unlearning_layer is None):
            return last_hidden
        B, T, H = last_hidden.shape
        x = last_hidden.reshape(-1, H)
        y = self.unlearning_layer(x)
        return y.reshape(B, T, H)

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
        tokenizer = self.processor  # type: ignore
        if isinstance(texts, str):
            texts = [texts]
        B = len(texts)
        if targets is None:
            enc = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )
            for k, v in enc.items():
                if isinstance(v, torch.Tensor):
                    enc[k] = v.to(self.device)
            return enc
        else:
            if isinstance(targets, str):
                targets = [targets]
            if len(targets) == 1 and B > 1:
                targets = [targets[0] for _ in range(B)]
            if len(targets) != B:
                raise ValueError(f"Targets and batch size mismatch: {len(targets)} vs {B}")
            # 逐样本构建 input_ids 与 labels，遮住 prompt 部分
            input_ids_list = []
            labels_list = []
            attn_list = []
            for t, y in zip(texts, targets):
                ids_prompt = tokenizer.encode(t, add_special_tokens=False)
                ids_target = tokenizer.encode(y, add_special_tokens=False)
                eos_id = tokenizer.eos_token_id
                if eos_id is not None:
                    full = ids_prompt + ids_target + [eos_id]
                else:
                    full = ids_prompt + ids_target
                input_ids_tensor = torch.tensor(full, dtype=torch.long)
                labels_tensor = input_ids_tensor.clone()
                labels_tensor[:len(ids_prompt)] = -100
                attn_tensor = torch.ones_like(labels_tensor, dtype=torch.long)
                input_ids_list.append(input_ids_tensor)
                labels_list.append(labels_tensor)
                attn_list.append(attn_tensor)
            max_len = max(x.size(0) for x in input_ids_list)
            def pad_to(x, val):
                if x.size(0) == max_len:
                    return x
                pad = torch.full((max_len - x.size(0),), val, dtype=x.dtype)
                return torch.cat([x, pad], dim=0)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
            input_ids = torch.stack([pad_to(x, pad_id) for x in input_ids_list], dim=0)
            labels = torch.stack([pad_to(x, -100) for x in labels_list], dim=0)
            attention_mask = torch.stack([pad_to(x, 0) for x in attn_list], dim=0)
            return {
                "input_ids": input_ids.to(self.device),
                "attention_mask": attention_mask.to(self.device),
                "labels": labels.to(self.device),
            }

    # 新增：将任意输入规格统一为“每条样本一个PIL列表”的批格式 List[List[PIL]]，并在训练时限制分辨率
    def _ensure_pil_per_sample(self, images) -> List[List]:
        """支持以下输入：
        - 单张图：PIL/tensor -> [[PIL]]
        - 单样本多图：List[PIL/tensor] -> [List[PIL]]
        - Batch：List[ Pils 或 List[PIL] ] -> List[List[PIL]]
        附加：若配置了 max_image_res，则按最长边缩放至不超过该分辨率。
        """
        def to_pil(x):
            return self._ensure_pil_list(x)[0] if not isinstance(x, list) else [self._ensure_pil_list(i)[0] for i in x]

        def maybe_resize(pil):
            max_res = int(self.max_image_res)
            if max_res <= 0:
                return pil
            try:
                w, h = pil.size
                scale = min(1.0, float(max_res) / float(max(w, h)))
                if scale < 1.0:
                    new_w = max(1, int(w * scale))
                    new_h = max(1, int(h * scale))
                    return pil.resize((new_w, new_h))
            except Exception:
                pass
            return pil

        def ensure_pil_list_and_resize(x):
            lst = self._ensure_pil_list(x)
            return [maybe_resize(i) for i in lst]

        if isinstance(images, list):
            # 判定是否为 batch（元素本身是列表或混合）
            if any(isinstance(el, list) for el in images):
                batch = []
                for el in images:
                    if isinstance(el, list):
                        batch.append(ensure_pil_list_and_resize(el))
                    else:
                        batch.append(ensure_pil_list_and_resize(el))  # 单图样本
                return batch
            else:
                # 单样本多图
                return [ensure_pil_list_and_resize(images)]
        else:
            # 单样本单图
            return [[ensure_pil_list_and_resize(images)[0]]]

    def _prepare_inputs(self, images, texts, targets: Optional[List[str]] = None):
        """准备模型输入；当提供targets时，构造labels与input_ids等长，并对prompt部分置-100。
        支持原生多图：images可以是 List[List[PIL]]（batch级），或 List[PIL]（单样本多图），或单图。
        文本-only模型将忽略 images。
        """
        # 文本-only 路径
        if self.text_only:
            return self._prepare_inputs_text_only(texts, targets)
        # logging.info(f"[KD] 文字样本数 {len(texts)} | 图片样本数 {len(images)} | 目标数 {len(targets)}")
        # texts 统一成列表
        if isinstance(texts, str):
            texts = [texts]
        # 图像规范化为“每条样本的图像列表”的批格式
        images_per_sample = self._ensure_pil_per_sample(images)  # List[List[PIL]]
        B = len(images_per_sample)
        logging.info(f"[KD] B样本数 {B}")

        # 文本广播/对齐
        if len(texts) == 1 and B > 1:
            texts = [texts[0] for _ in range(B)]
        if len(texts) != B:
            raise ValueError(f"Texts and images batch size mismatch: {len(texts)} vs {B}")

        # 构造仅prompt的对话（多次 {image} + text）
        convs_user_only = []
        for t, imgs in zip(texts, images_per_sample):
            content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": t}]
            convs_user_only.append([{ "role": "user", "content": content }])
        if targets is None:
            # 推理：只构造用户轮，添加generation prompt
            prompt_texts = self.processor.apply_chat_template(
                convs_user_only, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=prompt_texts, images=images_per_sample,
                return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len
            )
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            return inputs
        else:
            # 统一targets长度
            if isinstance(targets, str):
                targets = [targets]
            if len(targets) == 1 and B > 1:
                targets = [targets[0] for _ in range(B)]
            if len(targets) != B:
                raise ValueError(f"Targets and batch size mismatch: {len(targets)} vs {B}")
            # 1) 获取每条样本prompt的token长度（包含多图占位）
            prompt_token_ids_list = self.processor.apply_chat_template(
                convs_user_only, tokenize=True, add_generation_prompt=True
            )
            if isinstance(prompt_token_ids_list[0], int):
                prompt_token_ids_list = [prompt_token_ids_list]

            # 2) 构造包含assistant回复的完整对话
            convs_full = []
            for t, y, imgs in zip(texts, targets, images_per_sample):
                content_user = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": t}]
                convs_full.append([
                    {"role": "user", "content": content_user},
                    {"role": "assistant", "content": [{"type": "text", "text": y}]}
                ])

            full_texts = self.processor.apply_chat_template(
                convs_full, tokenize=False, add_generation_prompt=False
            )
            inputs = self.processor(
                text=full_texts,
                images=images_per_sample,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_seq_len,
            )
            # 3) 将 prompt 部分 labels 置 -100
            labels = inputs.get("labels", None)
            input_ids = inputs.get("input_ids", None)
            if (labels is None) and (input_ids is not None):
                labels = input_ids.clone()
            if labels is not None:
                for b_idx, prompt_ids in enumerate(prompt_token_ids_list):
                    n_prompt = len(prompt_ids)
                    labels[b_idx, :n_prompt] = -100
                inputs["labels"] = labels
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            return inputs

    def generate(self, images, texts: Union[str, List[str]], max_new_tokens: int = 128, do_sample: bool = True, temperature: float = 0.7):
        inputs = self._prepare_inputs(images, texts, targets=None)
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        if self.text_only:
            # 文本-only，直接解码
            tokenizer = self.processor  # type: ignore
            return tokenizer.batch_decode(out, skip_special_tokens=True)
        # 多模态
        return self.processor.batch_decode(out, skip_special_tokens=True)

    def compute_nll(self, images, texts: Union[str, List[str]], targets: Union[str, List[str]]):
        """返回负对数似然（越小越好）"""
        inputs = self._prepare_inputs(images, texts, targets)
        with torch.no_grad():
            out = self.model(**inputs)
            logits = out.logits  # [B, T, V]
            labels = inputs["labels"]  # [B, T]
            # 交叉熵：仅计算 labels != -100 的位置
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="mean")
        return loss

    def forward(self, images=None, texts: Union[str, List[str]] = None, targets: Union[str, List[str]] = None):
        """兼容旧训练接口：返回包含 loss 的对象（SimpleNamespace），以便 trainer 统一处理。"""
        if texts is None:
            raise ValueError("texts 不能为空")
        inputs = self._prepare_inputs(images, texts, targets)
        out = self.model(**inputs)
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
        out = self.model(**inputs)
        logits = out.logits  # [B, T, V]
        labels = inputs["labels"]  # [B, T]
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