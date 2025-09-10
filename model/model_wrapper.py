import torch
import torch.nn as nn
from types import SimpleNamespace
from typing import List, Union
from transformers import AutoModel, AutoProcessor
from model.unlearning_layer import UnlearningLayer

try:
    from torchvision.transforms.functional import to_pil_image
except Exception:
    to_pil_image = None


class QwenVLWithUnlearning(nn.Module):
    """
    使用真实 Qwen2.5-VL 作为冻结骨干，取其隐藏表示作为融合特征，插入遗忘层并接冻结分类头。
    与现有 trainer/eval 对接：forward/forward_teacher 返回包含 .logits 的对象，shape=[B, num_classes]
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", num_classes: int = 7, use_fast: bool = False):
        super().__init__()
        # 设备选择：优先使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1) 加载多模态模型与处理器（信任远程代码以启用自定义多模态前向）
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=use_fast)

        # 冻结骨干参数
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2) 设定融合表示维度（以 hidden_size 为准）
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            # 兜底：尝试常见字段或给一个保守默认
            hidden_size = getattr(self.backbone.config, "text_config", {}).get("hidden_size", 1024)
        self.hidden_size = int(hidden_size)

        # 3) 遗忘层 + 冻结分类头（输出与任务类别一致）
        self.unlearning_layer = UnlearningLayer(input_dim=self.hidden_size).to(self.device)
        self.classifier = nn.Linear(self.hidden_size, num_classes).to(self.device)
        for p in self.classifier.parameters():
            p.requires_grad = False

    def _ensure_pil_list(self, images: Union[torch.Tensor, "PIL.Image.Image", List]) -> List:
        """将输入统一为 PIL 图像列表，方便交给 AutoProcessor 处理。"""
        # 已是列表
        if isinstance(images, list):
            imgs = images
        else:
            imgs = [images]

        processed = []
        for img in imgs:
            if isinstance(img, torch.Tensor):
                # 形状: C,H,W；需要转换为PIL
                if to_pil_image is None:
                    # 简单兜底：转到CPU并转numpy后构造PIL（避免强依赖torchvision）
                    import numpy as np
                    from PIL import Image
                    t = img.detach().cpu()
                    if t.dim() == 3 and t.size(0) in (1, 3):
                        # C,H,W -> H,W,C
                        arr = t.numpy()
                        arr = (arr * 255.0).clip(0, 255).astype('uint8') if arr.max() <= 1.0 else arr.astype('uint8')
                        if arr.shape[0] == 1:
                            arr = np.repeat(arr, 3, axis=0)
                        arr = arr.transpose(1, 2, 0)
                        processed.append(Image.fromarray(arr))
                    else:
                        # 无法识别的形状，直接报错提示
                        raise ValueError("Unsupported tensor image shape; expected CxHxW with C=1 or 3.")
                else:
                    # 使用 torchvision 的 to_pil_image
                    t = img.detach().cpu()
                    if t.dim() == 3 and t.size(0) == 1:
                        t = t.repeat(3, 1, 1)  # 单通道扩展为3通道
                    processed.append(to_pil_image(t))
            else:
                # 假设已是 PIL.Image 或可被 processor 接受的类型
                processed.append(img)
        return processed

    def _prepare_inputs(self, images, texts, device: torch.device):
        # texts 统一成列表
        if isinstance(texts, str):
            texts = [texts]
        # 图像转为 PIL 列表
        image_list = self._ensure_pil_list(images)
        # 对齐长度：若文本为1条则广播到图像数
        if len(texts) == 1 and len(image_list) > 1:
            texts = [texts[0] for _ in range(len(image_list))]
        if len(texts) != len(image_list):
            raise ValueError(f"Texts and images count mismatch: {len(texts)} vs {len(image_list)}")

        # 为 Qwen2.5-VL 构造对话模板，插入图像占位符
        conversations = []
        for t in texts:
            conversations.append([
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": t}
                ]}
            ])
        # 生成包含视觉占位符的文本（区分单样本与批量）
        if len(texts) == 1:
            prompt_texts = [self.processor.apply_chat_template(conversations[0], tokenize=False, add_generation_prompt=False)]
        else:
            prompt_texts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)

        # 编码为模型输入（包含图像特征与带占位符的文本）
        inputs = self.processor(text=prompt_texts, images=image_list, return_tensors="pt", padding=True)
        # 移动到设备
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        return inputs

    def _compute_fused(self, images, texts) -> torch.Tensor:
        device = self.device
        inputs = self._prepare_inputs(images, texts, device)
        # 使用新的 autocast API（替换未来弃用的 torch.cuda.amp.autocast）
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = self.backbone(**inputs, output_hidden_states=True, return_dict=True)
        hidden_states = getattr(outputs, "hidden_states", None)
        last_hidden = None
        if hidden_states is not None and isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 0:
            last_hidden = hidden_states[-1]  # [B, L, H]
        else:
            last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise RuntimeError("Backbone did not return hidden states. Ensure trust_remote_code=True and correct processor/model pairing.")
        # 简单均值池化作为融合表示
        fused = last_hidden.mean(dim=1)  # [B, H]
        return fused

    def get_fused_features(self, images, texts):
        with torch.no_grad():
            return self._compute_fused(images, texts)

    def forward_teacher(self, images, texts):
        fused = self._compute_fused(images, texts)
        logits = self.classifier(fused)  # 教师：不经过遗忘层
        return SimpleNamespace(logits=logits)

    def forward(self, images, texts):
        fused = self._compute_fused(images, texts)
        filtered = self.unlearning_layer(fused)
        logits = self.classifier(filtered)
        return SimpleNamespace(logits=logits)