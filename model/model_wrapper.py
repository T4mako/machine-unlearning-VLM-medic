import logging

import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import List, Union, Optional
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoModelForImageTextToText
import torch.nn.functional as F
from config import config
from model.unlearning_layer import UnlearningLayer

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


class GenerativeQwenVLModel(nn.Module):
    """
    HuatuoGPT-Vision-7B 基于 Qwen2-7B 进行训练，使用 LLaVA-v1.5 架构
    用于KGA框架的image-text-to-text任务。
    支持：
    1. generate: 生成式推理
    2. compute_nll: 计算给定target的负对数似然（用于KGA知识差距计算）
    3. forward: 返回包含logits的对象以兼容旧接口
    另外：
    - 提供 loss_on_batch 用于蒸馏训练（带梯度）。
    """

    def __init__(self, model_name: str = config.model.model_name, use_fast: bool = config.model.use_fast):
        super().__init__()
        # 设备选择：优先使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = config.model.max_seq_len  # 针对对话式长文本设置上下文上限

        self.text_only: bool = False
        dtype = torch.bfloat16

        # 优先加载多模态生成模型
        self.model = None
        self.processor = None
        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=torch.bfloat16,
                device_map="auto",
                offload_folder="offload"
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
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map="auto",
                    offload_folder="offload",
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
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                    device_map="auto",
                    offload_folder="offload",
                )
                # 文本-only 使用 tokenizer 作为processor占位
                self.processor = AutoTokenizer.from_pretrained(
                    model_name,
                    use_fast=use_fast,
                    trust_remote_code=True,
                )
                logging.info("模型已加载（文本-only CausalLM）")

        # ===== 遗忘层 =====
        logging.info("初始化遗忘层...，但并未训练")
        self.hidden_size = int(getattr(self.model.config, "hidden_size", getattr(getattr(self.model, "config", object()), "hidden_size", 0)))
        self.unl_enabled: bool = bool(getattr(config.model, "enable_unl", False))
        self.unl_hidden_dim: int = int(getattr(config.model, "unl_hidden_dim", 128))
        self.unlearning_layer: Optional[UnlearningLayer] = None
        if self.unl_enabled and self.hidden_size > 0:
            self.unlearning_layer = UnlearningLayer(self.hidden_size, hidden_dim=self.unl_hidden_dim).to(self.device)

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

    # 新增：文本-only 输入准备
    def _prepare_inputs_text_only(self, texts, targets: Optional[List[str]] = None):
        if isinstance(texts, str):
            texts = [texts]
        B = len(texts)
        tokenizer = self.processor  # type: ignore
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
                # 在大多数CausalLM中，追加一个分隔符/空格更稳妥
                ids_target = tokenizer.encode(y, add_special_tokens=False)
                # 拼接，并在末尾追加 eos
                eos_id = tokenizer.eos_token_id
                if eos_id is not None:
                    full = ids_prompt + ids_target + [eos_id]
                else:
                    full = ids_prompt + ids_target
                input_ids_list.append(torch.tensor(full, dtype=torch.long))
                # labels：遮住 prompt 部分
                labels = torch.tensor(full, dtype=torch.long)
                labels[:len(ids_prompt)] = -100
                labels_list.append(labels)
                attn_list.append(torch.ones_like(labels, dtype=torch.long))
            # pad 到同长
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

    # 新增：将任意输入规格统一为“每条样本一个PIL列表”的批格式 List[List[PIL]]
    def _ensure_pil_per_sample(self, images) -> List[List]:
        """支持以下输入：
        - 单张图：PIL/tensor -> [[PIL]]
        - 单样本多图：List[PIL/tensor] -> [List[PIL]]
        - Batch：List[ Pils 或 List[PIL] ] -> List[List[PIL]]
        """
        def to_pil(x):
            return self._ensure_pil_list(x)[0] if not isinstance(x, list) else [self._ensure_pil_list(i)[0] for i in x]

        if isinstance(images, list):
            # 判定是否为 batch（元素本身是列表或混合）
            if any(isinstance(el, list) for el in images):
                batch = []
                for el in images:
                    if isinstance(el, list):
                        batch.append(self._ensure_pil_list(el))
                    else:
                        batch.append(self._ensure_pil_list(el))  # 单图样本
                return batch
            else:
                # 单样本多图
                return [self._ensure_pil_list(images)]
        else:
            # 单样本单图
            return [[self._ensure_pil_list(images)[0]]]

    def _prepare_inputs(self, images, texts, targets: Optional[List[str]] = None):
        """准备模型输入；当提供targets时，构造labels与input_ids等长，并对prompt部分置-100。
        支持原生多图：images可以是 List[List[PIL]]（batch级），或 List[PIL]（单样本多图），或单图。
        文本-only模型将忽略 images。
        """
        # 文本-only 路径
        if self.text_only:
            return self._prepare_inputs_text_only(texts, targets)
        logging.info(f"[KD] 文字样本数 {len(texts)} | 图片样本数 {len(images)} | 目标数 {len(targets)}")
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
        logging.info(f"[convs_user_only] {convs_user_only}")
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
            logging.info(f"[KD] B样本数 {B} | 目标数 {len(targets)}")
            # 1) 获取每条样本prompt的token长度（包含多图占位）
            logging.info(f"convs_user_only {convs_user_only}")
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
                text=full_texts, images=images_per_sample,
                return_tensors="pt", padding=True, truncation=True, max_length=self.max_seq_len
            )
            # 构造labels：遮住prompt部分
            input_ids = inputs["input_ids"]
            labels = torch.full_like(input_ids, -100)
            pad_id = self.processor.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.processor.tokenizer.eos_token_id
            for i, pids in enumerate(prompt_token_ids_list):
                prompt_len = len(pids)
                row = input_ids[i]
                non_pad = (row != pad_id).nonzero(as_tuple=False).squeeze(-1)
                if non_pad.numel() == 0:
                    continue
                last_valid = int(non_pad[-1])
                start = min(prompt_len, last_valid + 1)
                labels[i, start:last_valid + 1] = row[start:last_valid + 1]
            inputs["labels"] = labels
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)
            return inputs

    def generate(self, images, texts, max_length: int = 100, temperature: float = 0.7, do_sample: bool = True):
        """生成式推理，支持原生多图输入；文本-only 模型忽略 images。"""
        self.model.eval()
        if self.text_only:
            if isinstance(texts, str):
                texts = [texts]
            tokenizer = self.processor  # type: ignore
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
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **enc,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            return decoded

        # 多模态路径
        # 构造用户轮对话模板（多图）
        if isinstance(texts, str):
            texts = [texts]
        images_per_sample = self._ensure_pil_per_sample(images)
        B = len(images_per_sample)
        if len(texts) == 1 and B > 1:
            texts = [texts[0] for _ in range(B)]
        if len(texts) != B:
            raise ValueError(f"Texts and images batch size mismatch: {len(texts)} vs {B}")

        convs_user_only = []
        for t, imgs in zip(texts, images_per_sample):
            content = [{"type": "image"} for _ in imgs] + [{"type": "text", "text": t}]
            convs_user_only.append([{ "role": "user", "content": content }])

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
        
        with torch.no_grad():
            # 生成阶段不应用遗忘层，保持原生生成行为（如有需要可开启）
            generated_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # 解码生成的文本（去除输入部分，按每条样本的prompt长度切分更稳妥）
        prompt_token_ids_list = self.processor.apply_chat_template(
            convs_user_only, tokenize=True, add_generation_prompt=True
        )
        if isinstance(prompt_token_ids_list[0], int):
            prompt_token_ids_list = [prompt_token_ids_list]
        decoded = []
        for i in range(generated_ids.size(0)):
            start = len(prompt_token_ids_list[i])
            text = self.processor.tokenizer.decode(
                generated_ids[i, start:], skip_special_tokens=True
            )
            decoded.append(text)
        return decoded

    def compute_nll(self, images, texts, targets):
        """计算给定target的负对数似然，用于KGA/EUL知识差距计算。若开启遗忘层，则基于遗忘层后的logits计算。"""
        self.model.eval()
        inputs = self._prepare_inputs(images, texts, targets)
        
        with torch.no_grad():
            if self.text_only:
                outputs = self.model(**inputs)
                loss = outputs.loss
            else:
                if self.unl_enabled and (self.unlearning_layer is not None):
                    outputs = self.model(**inputs, output_hidden_states=True)
                    last_hidden = outputs.hidden_states[-1]
                    last_hidden = self._apply_unl(last_hidden)
                    lm_head = self.model.get_output_embeddings()
                    logits = lm_head(last_hidden)
                    labels = inputs["labels"]
                    # 与HF一致：shift一位
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                else:
                    outputs = self.model(**inputs)
                    loss = outputs.loss  # 已经是NLL
        return loss

    # 新增：训练用，返回带梯度的loss
    def loss_on_batch(self, images, texts, targets):
        inputs = self._prepare_inputs(images, texts, targets)
        if self.text_only:
            outputs = self.model(**inputs)
            return outputs.loss
        else:
            if self.unl_enabled and (self.unlearning_layer is not None):
                outputs = self.model(**inputs, output_hidden_states=True)
                last_hidden = outputs.hidden_states[-1]
                last_hidden = self._apply_unl(last_hidden)
                lm_head = self.model.get_output_embeddings()
                logits = lm_head(last_hidden)
                labels = inputs["labels"]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100
                )
                return loss
            else:
                outputs = self.model(**inputs)
                return outputs.loss

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