"""
Color head using Qwen3-VL. A head implements

    predict(images: list[PIL.Image]) -> list[(color: str, confidence: float)]

Colors come from COLORS. Qwen3-VL zero-shot answers are scored over the fixed color list
(next-token probability among the options), so the output is always a valid color and the
confidence is a real probability.
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image

COLORS = ["white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "brown", "beige", "gold", "other"]

HEADS: Dict[str, Callable[..., "ColorHead"]] = {}


def register(name: str):
    def deco(cls):
        HEADS[name] = cls
        return cls
    return deco


def build_head(name: str, **kwargs) -> "ColorHead":
    if name not in HEADS:
        raise ValueError(f"Unknown color head '{name}'. Available: {sorted(HEADS)}")
    return HEADS[name](**kwargs)


class ColorHead:
    def __init__(self, **_):
        pass

    def predict(self, images: List[Image.Image]) -> List[Tuple[str, float]]:
        raise NotImplementedError


@register("qwen3vl")
class Qwen3VLHead(ColorHead):
    PROMPT = ("What is the main body color of the vehicle in this image? Answer with exactly one word from this list: "
              + ", ".join(COLORS) + ".")

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda", max_side: int = 336,
                 load_in_4bit: bool = True, **_):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError as e:
            raise ImportError("Qwen3-VL needs a recent transformers with Qwen3VL support: pip install -U 'transformers>=4.57' accelerate") from e

        self.torch, self.device, self.max_side = torch, device, max_side
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"

        load_kw = {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}
        if load_in_4bit and device == "cuda" and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                load_kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
                load_kw["device_map"] = "auto"
                print(f"[color] 4-bit NF4 quantization enabled (bitsandbytes)")
            except Exception as e:
                print(f"[color] 4-bit quantization failed ({e}), falling back to bfloat16")
                load_kw["device_map"] = "auto"
        elif device == "cuda" and torch.cuda.is_available():
            load_kw["device_map"] = "auto"

        try:
            self.model = ModelCls.from_pretrained(model_id, **load_kw)
        except TypeError:
            load_kw.pop("attn_implementation", None)
            load_kw["dtype"] = load_kw.pop("torch_dtype", torch.bfloat16)
            self.model = ModelCls.from_pretrained(model_id, **load_kw)

        if not hasattr(self.model, "hf_device_map"):
            self.model.to(device)
        self.model.eval()

        tok = self.processor.tokenizer
        self.first_ids = []
        for c in COLORS:
            ids = tok.encode(c, add_special_tokens=False)
            self.first_ids.append(ids[0])
        if len(set(self.first_ids)) != len(COLORS):
            raise RuntimeError(f"Color words share a first token, scoring would be ambiguous: {self.first_ids}")
        mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        print(f"[color] Qwen3-VL loaded from {model_id} on {device} (VRAM: {mem:.2f} GB, max_side={max_side})")

    def predict(self, images):
        torch = self.torch
        msgs = []
        for im in images:
            if max(im.size) > self.max_side:
                s = self.max_side / max(im.size)
                im = im.resize((max(28, round(im.width * s)), max(28, round(im.height * s))), Image.BILINEAR)
            msgs.append([{"role": "user", "content": [{"type": "image", "image": im},
                                                      {"type": "text", "text": self.PROMPT}]}])
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :].float()
        probs = torch.softmax(logits[:, self.first_ids], dim=-1).cpu().numpy()
        return [(COLORS[int(p.argmax())], float(p.max())) for p in probs]
