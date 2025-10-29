
from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
import torch
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

def generate_single_with_stats(
    model: LlavaOnevisionForConditionalGeneration,
    processor: LlavaOnevisionProcessor,
    images_for_prompt: List[Image.Image],
    model_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
) -> Tuple[str | None, Dict[str, Any]]:
    # 1) Preprocess on CPU
    t0 = time.perf_counter()
    raw_inputs = processor(
        images=[images_for_prompt],   # nested list -> ONE sample with many images
        text=[model_prompt],          # list -> batch size 1
        padding=True,
        return_tensors="pt",
    )
    t1 = time.perf_counter()
    prep_time_s = t1 - t0

    input_ids = raw_inputs["input_ids"]
    assert input_ids.shape[0] == 1, "Multi-image mode should create batch size 1; use images=[[...]] and text=[...]."
    pixel_values = raw_inputs["pixel_values"]
    image_sizes = raw_inputs["image_sizes"]
    expanded_prompt_tokens = int(input_ids.shape[1])
    total_tiles_in_batch = int(pixel_values.shape[0])
    pixel_values_shape = tuple(pixel_values.shape)
    image_sizes_shape = tuple(image_sizes.shape)

    # tokens before <image> expansion
    pre_ids = processor.tokenizer(model_prompt, return_tensors="pt")["input_ids"][0]
    pre_expansion_text_tokens = int(pre_ids.numel())

    # tiling per-image (for sanity)
    tiling_stats = analyze_image_tiling(processor, images_for_prompt)
    tokens_per_tile = tiling_stats["tokens_per_tile"]
    expected_total_tiles = tiling_stats["total_tiles_individual"]
    expected_total_vision_tokens = tiling_stats["total_vision_tokens_individual"]
    actual_total_vision_tokens = int(total_tiles_in_batch * tokens_per_tile)

    # 2) Move to device
    device = next(model.parameters()).device
    t2 = time.perf_counter()
    inputs = _move_inputs_to_device_half_if_cuda(raw_inputs, device)
    t3 = time.perf_counter()
    move_time_s = t3 - t2

    # 3) Generate
    gen_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": do_sample}
    if do_sample:
        gen_kwargs["top_p"] = top_p

    oom = False
    output_tokens = None
    gen_time_s = None
    decoded_answer = None
    new_tokens_generated = None
    tps = None

    try:
        with torch.no_grad():
            g0 = time.perf_counter()
            output_tokens = model.generate(**inputs, **gen_kwargs)
            g1 = time.perf_counter()
        gen_time_s = g1 - g0
        total_len = int(output_tokens.shape[1])
        prompt_len = int(inputs["input_ids"].shape[1])
        new_tokens_generated = total_len - prompt_len
        tps = (new_tokens_generated / gen_time_s) if gen_time_s else None

        decoded_full = processor.batch_decode(output_tokens, skip_special_tokens=True)[0]
        decoded_answer = _strip_prefix(decoded_full)

    except torch.cuda.OutOfMemoryError:
        oom = True

    stats: Dict[str, Any] = {
        "num_images": len(images_for_prompt),
        "per_image_tiles": tiling_stats["per_image"],
        "expected_total_tiles_individual": expected_total_tiles,
        "expected_total_vision_tokens_individual": expected_total_vision_tokens,
        "pixel_values_shape": pixel_values_shape,
        "image_sizes_shape": image_sizes_shape,
        "actual_total_tiles_in_batch": total_tiles_in_batch,
        "actual_total_vision_tokens_in_batch": actual_total_vision_tokens,
        "tokens_per_tile": tokens_per_tile,
        "pre_expansion_text_tokens": pre_expansion_text_tokens,
        "prompt_tokens_after_expansion": expanded_prompt_tokens,
        "new_tokens_generated": new_tokens_generated,
        "tokens_per_second": tps,
        "prep_time_s": prep_time_s,
        "move_to_device_time_s": move_time_s,
        "generate_time_s": gen_time_s,
        "total_time_s": prep_time_s + move_time_s + (gen_time_s or 0.0),
        "oom": oom,
    }
    return decoded_answer, stats
