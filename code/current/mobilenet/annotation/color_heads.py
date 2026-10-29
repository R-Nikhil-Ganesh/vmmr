"""
Color head using Qwen3-VL. A head implements

    predict(images: list[PIL.Image]) -> list[(color: str, confidence: float)]

Colors come from COLORS. Qwen3-VL zero-shot answers are scored over the fixed color list
(next-token probability among the options), so the output is always a valid color and the
confidence is a real probability.

Heads may also implement predict_proba(images) -> (probs N x len(COLORS), mass N), the full
distribution over COLORS plus the raw next-token probability mass that fell on the options
(low mass = the model wanted to answer something else). eval_color_teacher.py uses it.
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


PROMPTS = {
    "short": ("What is the main body color of the vehicle in this image? Answer with exactly one word from this list: "
              + ", ".join(COLORS) + "."),
    "defined": ("What is the paint color of the vehicle's body panels in this image? Judge only the painted body "
                "(doors, hood, fenders). Ignore windows, tyres, bumpers, trim, reflections, shadows and the color of "
                "the lighting.\n"
                "Definitions: white = bright, flat, not metallic. silver = light metallic gray with a shine. "
                "gray = medium or dark gray. black = very dark with no visible hue; dark blue, dark green and dark "
                "red cars are blue, green and red, not black. beige = pale warm cream or champagne. gold = warm "
                "metallic yellow-brown. brown = brown or bronze, darker than beige. orange, yellow, red, blue, green "
                "as usual. other = any color not listed (purple, pink, multi-color wraps).\n"
                "Answer with exactly one word from this list: " + ", ".join(COLORS) + "."),
}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:len(COLORS)]


@register("qwen3vl")
class Qwen3VLHead(ColorHead):
    """
    Options (defaults reproduce Annotation Runs 1/2):
      scoring      "first_token": softmax over the first token of each lowercase color word (original).
                   "variants":    full-vocab softmax, summed over the first tokens of "white", "White", " white", " White".
                   "letters":     options listed as "A) white ... M) other", scored on the letter tokens.
      prompt_style key of PROMPTS ("short" = original, "defined" = with color definitions).
      center_crop  keep the central fraction of each image (0 = off) before resizing to max_side.
      tta          "none" or "flip" (average probabilities over the image and its mirror).
    """
    PROMPT = PROMPTS["short"]

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct", device: str = "cuda", max_side: int = 336,
                 load_in_4bit: bool = True, scoring: str = "first_token", prompt_style: str = "short",
                 center_crop: float = 0.0, tta: str = "none", **_):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import Qwen3VLForConditionalGeneration as ModelCls
        except ImportError as e:
            raise ImportError("Qwen3-VL needs a recent transformers with Qwen3VL support: pip install -U 'transformers>=4.57' accelerate") from e

        if scoring not in ("first_token", "variants", "letters"):
            raise ValueError(f"scoring must be first_token, variants or letters, got {scoring!r}")
        if prompt_style not in PROMPTS:
            raise ValueError(f"prompt_style must be one of {sorted(PROMPTS)}, got {prompt_style!r}")
        if tta not in ("none", "flip"):
            raise ValueError(f"tta must be none or flip, got {tta!r}")
        if not 0.0 <= center_crop < 1.0:
            raise ValueError(f"center_crop must be in [0, 1), got {center_crop}")
        self.torch, self.device, self.max_side = torch, device, max_side
        self.scoring, self.center_crop, self.tta = scoring, center_crop, tta
        self.prompt = PROMPTS[prompt_style]
        if scoring == "letters":
            options = "\n".join(f"{l}) {c}" for l, c in zip(LETTERS, COLORS))
            self.prompt = (self.prompt.rsplit("Answer with", 1)[0]
                           + "Options:\n" + options + "\nAnswer with the letter only.")
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
        # Token sets per option for "variants"/"letters": first token of each surface form, deduplicated
        words = LETTERS if scoring == "letters" else COLORS
        forms = (lambda w: [w, " " + w]) if scoring == "letters" else (lambda w: [w, w.capitalize(), " " + w, " " + w.capitalize()])
        self.option_ids = [sorted({tok.encode(f, add_special_tokens=False)[0] for f in forms(w)}) for w in words]
        flat = [i for ids in self.option_ids for i in ids]
        if scoring != "first_token" and len(flat) != len(set(flat)):
            raise RuntimeError(f"Options share a first token under scoring={scoring}: {self.option_ids}")
        mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
        print(f"[color] Qwen3-VL loaded from {model_id} on {device} (VRAM: {mem:.2f} GB, max_side={max_side}, "
              f"scoring={scoring}, prompt={prompt_style}, center_crop={center_crop}, tta={tta})")

    def _prep(self, im):
        if self.center_crop:
            dx, dy = round(im.width * self.center_crop / 2), round(im.height * self.center_crop / 2)
            im = im.crop((dx, dy, im.width - dx, im.height - dy))
        if max(im.size) > self.max_side:
            s = self.max_side / max(im.size)
            im = im.resize((max(28, round(im.width * s)), max(28, round(im.height * s))), Image.BILINEAR)
        return im

    def _score(self, images):
        torch = self.torch
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                              {"type": "text", "text": self.prompt}]}] for im in images]
        inputs = self.processor.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt", padding=True)
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[:, -1, :].float()
        full = torch.softmax(logits, dim=-1)
        if self.scoring == "first_token":
            opt = full[:, self.first_ids]
            probs = torch.softmax(logits[:, self.first_ids], dim=-1)
        else:
            opt = torch.stack([full[:, ids].sum(-1) for ids in self.option_ids], dim=-1)
            probs = opt / opt.sum(-1, keepdim=True).clamp_min(1e-12)
        return probs.cpu().numpy(), opt.sum(-1).cpu().numpy()

    def predict_proba(self, images):
        images = [self._prep(im) for im in images]
        probs, mass = self._score(images)
        if self.tta == "flip":
            p2, m2 = self._score([im.transpose(Image.FLIP_LEFT_RIGHT) for im in images])
            probs, mass = (probs + p2) / 2, (mass + m2) / 2
        return probs, mass

    def predict(self, images):
        probs, _ = self.predict_proba(images)
        return [(COLORS[int(p.argmax())], float(p.max())) for p in probs]
