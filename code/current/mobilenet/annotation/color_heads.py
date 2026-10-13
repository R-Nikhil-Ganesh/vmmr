"""
Pluggable color heads. A head implements

    predict(images: list[PIL.Image]) -> list[(color: str, confidence: float)]

and is registered under a name so annotate.py can pick one with --color-head. Colors come from COLORS.

  classical   no model, mean HSV of the central region: a cheap baseline (and the local test stub)
  qwen3vl     Qwen3-VL zero-shot; the answer is scored over the fixed color list (next-token probability among
              the options), so the output is always a valid color and the confidence is a real probability
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


@register("classical")
class ClassicalHead(ColorHead):
    """Mean color of the central 50% of the crop, mapped to the nearest named color. Weak (reflections, plates,
    background), kept only as a baseline."""

    # name -> (h in [0,360), s, v) prototypes
    _PROTOS = {
        "white": (0, 0.03, 0.95), "black": (0, 0.05, 0.10), "silver": (0, 0.03, 0.72), "gray": (0, 0.03, 0.45),
        "red": (0, 0.85, 0.65), "blue": (220, 0.8, 0.55), "green": (120, 0.7, 0.45), "yellow": (55, 0.85, 0.9),
        "orange": (28, 0.9, 0.9), "brown": (25, 0.7, 0.35), "beige": (40, 0.25, 0.8), "gold": (45, 0.6, 0.7),
    }

    def predict(self, images):
        out = []
        for im in images:
            w, h = im.size
            c = im.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)).resize((32, 32))
            hsv = np.array(c.convert("HSV"), dtype=np.float32).reshape(-1, 3)
            hh, ss, vv = hsv[:, 0] * 360 / 255, hsv[:, 1] / 255, hsv[:, 2] / 255
            best, scores = None, {}
            for name, (ph, ps, pv) in self._PROTOS.items():
                dh = np.minimum(np.abs(hh - ph), 360 - np.abs(hh - ph)) / 180
                wgt = np.minimum(ss, ps) + 0.05                      # hue only matters for saturated colors
                d = (wgt * dh) ** 2 + (ss - ps) ** 2 + (vv - pv) ** 2
                scores[name] = float(np.exp(-d.mean() * 8))
            best = max(scores, key=scores.get)
            tot = sum(scores.values()) or 1.0
            out.append((best, scores[best] / tot))
        return out


@register("qwen3vl")
class Qwen3VLHead(ColorHead):
    PROMPT = ("What is the main body color of the vehicle in this image? Answer with exactly one word from this list: "
              + ", ".join(COLORS) + ".")

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-4B-Instruct", device: str = "cuda", max_side: int = 448, **_):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError as e:
            raise ImportError("Qwen3-VL needs a recent transformers: pip install -U 'transformers>=4.57' accelerate") from e

        self.torch, self.device, self.max_side = torch, device, max_side
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.tokenizer.padding_side = "left"
        try:
            self.model = ModelCls.from_pretrained(model_id, dtype=torch.bfloat16)
        except TypeError:
            self.model = ModelCls.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.model.to(device).eval()

        tok = self.processor.tokenizer
        self.first_ids = []
        for c in COLORS:
            ids = tok.encode(c, add_special_tokens=False)
            self.first_ids.append(ids[0])
        if len(set(self.first_ids)) != len(COLORS):
            raise RuntimeError(f"Color words share a first token, scoring would be ambiguous: {self.first_ids}")
        print(f"[color] Qwen3-VL loaded from {model_id} on {device}")

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
