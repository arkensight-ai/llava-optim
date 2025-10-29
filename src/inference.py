
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import torch
from benchmark import PhaseTimer
import time
from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

def _maybe_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    name = str(name).lower()
    if name in ("fp16", "float16", "torch.float16"):
        return torch.float16
    if name in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32", "torch.float32"):
        return torch.float32
    return None

def load_model(
    model_id: str,
    four_bit: Optional[bool] = None,               # legacy path (still works)
    quant: Optional[Dict[str, Any]] = None,        # new Hydra path
) -> Tuple[LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor]:

    # If `quant` provided (Hydra), use it; otherwise keep legacy behavior
    if quant is None:
        quant = {}
        if four_bit:
            quant.update(dict(
                name="bnb4_nf4",
                load_in_4bit=True,
                bnb_4bit_compute_dtype="fp16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ))
        else:
            quant.update(dict(name="fp16", dtype="fp16"))

    name = str(quant.get("name", "fp16")).lower()
    dtype = _maybe_dtype(quant.get("dtype"))

    # Build kwargs for from_pretrained
    from_kwargs: Dict[str, Any] = dict(device_map="auto")

    if name.startswith("bnb4") or quant.get("load_in_4bit", False):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_maybe_dtype(quant.get("bnb_4bit_compute_dtype")) or torch.bfloat16,
            bnb_4bit_use_double_quant=bool(quant.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=str(quant.get("bnb_4bit_quant_type", "nf4")),
        )
        from_kwargs["quantization_config"] = bnb
    elif name in ("bnb8", "int8") or quant.get("load_in_8bit", False):
        from_kwargs["load_in_8bit"] = True
    else:
        # default: fp16/bf16/fp32
        if dtype is None:
            dtype = torch.float16
        from_kwargs["dtype"] = dtype

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        **from_kwargs,
    )
    processor = LlavaOnevisionProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor

def _move_inputs_to_device_half_if_cuda(
    inputs: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if device.type == "cuda" and v.dtype == torch.float32:
                v = v.to(dtype=torch.float16)
        out[k] = v
    return out

def _strip_prefix(decoded: str) -> str:
    token = "assistant"
    if token in decoded:
        decoded = decoded.split(token, 1)[1]
    return decoded.strip()

def analyze_image_tiling(
    processor: LlavaOnevisionProcessor,
    images_for_prompt: List[Image.Image],
) -> Dict[str, Any]:
    """Count tiles per image using the processor's image_processor."""
    tokens_per_tile = getattr(processor, "tokens_per_tile", 256)
    per_image_info: List[Dict[str, Any]] = []
    total_tiles = 0

    for idx, img in enumerate(images_for_prompt):
        single_out = processor.image_processor(images=[img])
        pv = single_out["pixel_values"]
        n_tiles = int(pv.shape[0]) if isinstance(pv, torch.Tensor) else len(pv)
        per_image_info.append(
            {
                "image_index": idx,
                "orig_size_wh": img.size,
                "tiles": n_tiles,
                "vision_tokens": int(n_tiles * tokens_per_tile),
            }
        )
        total_tiles += n_tiles

    return {
        "tokens_per_tile": tokens_per_tile,
        "per_image": per_image_info,
        "total_tiles_individual": int(total_tiles),
        "total_vision_tokens_individual": int(total_tiles * tokens_per_tile),
    }

@torch.inference_mode()
def generate_single_with_stats(
    model,
    processor,
    images_for_prompt: List[Any],
    model_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (pred, stats) where stats contains per-phase timings and token counts.
    """
    device = model.device if hasattr(model, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timer = PhaseTimer()

    # ENCODE (processor -> tensors)
    timer.start("encode")
    enc = processor(
        images=images_for_prompt,
        text=model_prompt,
        return_tensors="pt",
        padding=True,
    )
    input_tokens = int(enc["input_ids"].shape[-1])
    enc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
    timer.stop("encode")

    # GENERATE
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timer.start("generate")
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
    )
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    timer.stop("generate")

    # DECODE
    timer.start("decode")
    # safer to use tokenizer directly
    pred = processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
    timer.stop("decode")

    output_tokens = int(out.shape[-1] - input_tokens)
    total_s = sum(timer.elapsed_s.get(p, 0.0) for p in ("encode", "generate", "decode"))
    gen_s = timer.elapsed_s.get("generate", 0.0)
    tps = (output_tokens / gen_s) if gen_s > 0 else float("nan")

    stats = {
        "time_s": dict(timer.elapsed_s),
        "n_images": len(images_for_prompt),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "t_total_s": total_s,
        "tokens_per_s": tps,
    }
    return pred, stats
