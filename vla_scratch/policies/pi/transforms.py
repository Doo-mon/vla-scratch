import torch
import torch.nn.functional as F
import importlib
from typing import Dict, List


from vla_scratch.transforms.base import TransformFn
from vla_scratch.transforms.data_keys import (
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    TASK_KEY,
    TOKENIZED_KEY,
    TOKENIZED_MASK_KEY,
)


class PreprocessImage(TransformFn):
    def __init__(self, target_size: tuple[int, int] = (224, 224)) -> None:
        self.target_size = target_size

    def compute(self, sample: Dict) -> Dict:
        images: torch.Tensor = sample[PROCESSED_IMAGE_KEY].to(torch.float32)
        images = F.interpolate(
            images,
            size=self.target_size,
            mode="bilinear",
            align_corners=False,
        )
        images = (images / 255.0 - 0.5) / 0.5
        sample[PROCESSED_IMAGE_KEY] = images
        if PROCESSED_IMAGE_MASK_KEY not in sample:
            mask = torch.ones(
                (images.shape[0], 1), dtype=torch.bool, device=images.device
            )
            sample[PROCESSED_IMAGE_MASK_KEY] = mask
        return sample


class StructurePrompt(TransformFn):
    """Format the task string into a language prompt."""

    def compute(self, sample: Dict) -> Dict:
        task_prompt: str = sample[TASK_KEY]
        sample["prompt"] = f"<bos>Task: {task_prompt}; \n Action:"
        return sample


class TokenizePrompt(TransformFn):
    def __init__(
        self,
        processor_class: str,
        model_id: str,
        max_length: int = 256,
    ) -> None:
        processors = importlib.import_module("transformers")
        processor_cls = getattr(processors, processor_class)
        processor = processor_cls.from_pretrained(model_id)
        # from transformers import PaliGemmaProcessor
        # processor = PaliGemmaProcessor.from_pretrained(model_id)
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

    def compute(self, sample: Dict) -> Dict:
        prompt: str = sample["prompt"]
        encoded = self.tokenizer(
            prompt,
            max_length=self.max_length,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        sample[TOKENIZED_KEY] = encoded["input_ids"].squeeze(0).long()
        sample[TOKENIZED_MASK_KEY] = encoded["attention_mask"].squeeze(0).bool()
        return sample


class QwenChatTemplateTokenize(TransformFn):
    """Tokenize prompt using Qwen3-VL chat template.

    This uses `AutoProcessor.apply_chat_template` to build a chat-formatted prompt that
    includes image placeholders, then tokenizes to fixed length for batching.

    Note: This prepares only text token IDs and masks. Qwen3-VL models also expect
    processed vision inputs at inference/training time, which are not produced here.
    """

    def __init__(
        self,
        model_id: str,
        max_length: int = 256,
        add_generation_prompt: bool = True,
        padding: str | bool = "max_length",
    ) -> None:
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        # Qwen chat models commonly use left-padding for generation
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"
        self.max_length = max_length
        self.add_generation_prompt = add_generation_prompt
        self.padding = padding

    def compute(self, sample: Dict) -> Dict:
        # Expect prior StructurePrompt to set `sample["prompt"]`
        prompt: str = sample.get("prompt", sample.get(TASK_KEY, ""))
        images: torch.Tensor = sample[PROCESSED_IMAGE_KEY]

        # Build messages with image content then the user prompt
        # images: [n_cam, 3, H, W] (torch.Tensor)
        content: List[Dict] = [
            {"type": "image", "image": img} for img in images
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        # Produce chat-formatted string, then tokenize to fixed length
        chat_str = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=self.add_generation_prompt,
        )
        encoded = self.processor.tokenizer(
            chat_str,
            max_length=self.max_length,
            truncation=True,
            padding=self.padding,
            return_tensors="pt",
        )

        sample[TOKENIZED_KEY] = encoded["input_ids"].squeeze(0).long()
        sample[TOKENIZED_MASK_KEY] = encoded["attention_mask"].squeeze(0).bool()
        return sample
