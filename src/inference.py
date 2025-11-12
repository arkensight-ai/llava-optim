
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Union
import torch
from benchmark import PhaseTimer, build_generation_stats
from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)
from debug_utils import DebugController, count_tiles_from_pixel_values, debug_seq_report

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
    quant: Optional[Dict[str, Any]] = None,
) -> Tuple[LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor]:

    ####### Quantization handling #######

    if quant is None:
        print("[Cfg] No quantization specified, defaulting to fp16.")
        quant = {}
        quant.update(dict(name="fp16", dtype="fp16"))

    name = str(quant.get("name", "fp16")).lower()
    dtype = _maybe_dtype(quant.get("dtype"))

    # Build kwargs for from_pretrained
    from_kwargs: Dict[str, Any] = dict(device_map="auto")

    if name.startswith("bnb4") or quant.get("load_in_4bit", False):
        print("[Cfg] Loading model with 4-bit quantization.")
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
    processor = LlavaOnevisionProcessor.from_pretrained(model_id, use_fast=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model.eval()
    return model, processor

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

def _token_lengths(input_ids: torch.Tensor, pad_id: int) -> List[int]:
    # counts non-pad tokens per row
    return (input_ids != pad_id).sum(dim=1).tolist()

def _normalize_generation_inputs(
    images_input: Union[List[Any], List[List[Any]]],
    prompts_input: Union[str, List[str]],
) -> Tuple[List[List[Any]], List[str], bool]:
    if isinstance(prompts_input, str):
        if not isinstance(images_input, list):
            raise TypeError("images must be a list of images when passing a single prompt.")
        return [images_input], [prompts_input], True
    if not isinstance(prompts_input, list):
        raise TypeError("model_prompts must be a string or a list of strings.")
    if not isinstance(images_input, list):
        raise TypeError("images must be a list.")
    if len(images_input) != len(prompts_input):
        raise ValueError("images and prompts must align.")
    if len(prompts_input) == 0:
        return [], [], False
    if not all(isinstance(sample, list) for sample in images_input):
        raise TypeError("For batched generation, images must be a list of image lists.")
    n_imgs = len(images_input[0])
    if not all(len(sample) == n_imgs for sample in images_input):
        raise ValueError("For simple batching, all samples must have the same number of images.")
    return images_input, prompts_input, False

@torch.inference_mode()
def generate_with_stats(
    model,
    processor,
    images: Union[List[Any], List[List["PIL.Image.Image"]]],
    model_prompts: Union[str, List[str]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    top_p: float = 0.9,
    temperature: float = 1.0,
    dbg: Optional[DebugController] = None,
):
    """
    Unified generation helper that handles both single-sample and batched inference.
    Pass a single prompt (str) with a list of images for single-mode,
    or lists of prompts/images for batch-mode.
    """
    dbg = dbg or DebugController(None)

    images_batch, prompts_batch, single_mode = _normalize_generation_inputs(images, model_prompts)
    bs = len(prompts_batch)
    if bs == 0:
        return ("", {}) if single_mode else ([], [])

    timer = PhaseTimer()

    @timer.measure("encode")
    def _encode_batch():
        enc_local = processor(
            text=prompts_batch,
            images=images_batch,           # list[list[PIL.Image]]
            padding=True,                  # pad text to max length in the batch
            return_tensors="pt",
        )
        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return enc_local.to(device)

    enc = _encode_batch()

    if dbg.enabled:
        pv = enc.get("pixel_values", None)
        tiles = count_tiles_from_pixel_values(pv)

        try:
            per_tile = getattr(model.config, "vision_feature_tokens", 256)
        except Exception:
            per_tile = 256
        vision_tokens = tiles * per_tile

        cfg_txt = getattr(model.config, "text_config", model.config)
        hidden = getattr(cfg_txt, "hidden_size", None)
        heads  = getattr(cfg_txt, "num_attention_heads", None)
        layers = getattr(cfg_txt, "num_hidden_layers", None)


        # len of the longest (padded) text row
        text_len = int(enc["input_ids"].shape[-1]) if "input_ids" in enc else None

        debug_seq_report(
            dbg,
            provider="torch",
            text_len=text_len,
            vision_tokens=vision_tokens,
            tiles=tiles,
            hidden_size=hidden,
            heads=heads,
            layers=layers,
        )

    @timer.measure("generate")
    def _generate_batch():
        return model.generate(
            **enc,
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
        )

    gen_out = _generate_batch()

    if dbg.enabled and dbg.detail.gen_summary:
        out_ids = gen_out[0] if hasattr(gen_out, "__array__") else gen_out.sequences[0]
        prompt_len = int(enc["input_ids"].shape[-1])
        gen_len = int(out_ids.shape[-1]) - prompt_len
        print(f"[debug/torch] generated_tokens={gen_len} (prompt={prompt_len} → total={int(out_ids.shape[-1])})")


    @timer.measure("decode")
    def _decode_batch():
        pad_id = processor.tokenizer.pad_token_id
        input_lens = _token_lengths(enc["input_ids"], pad_id=pad_id)  # per-sample prompt lengths
        preds_local: list[str] = []
        inp_tok_local: list[int] = []
        out_tok_local: list[int] = []

        for i, L in enumerate(input_lens):
            new_tokens = gen_out[i, L:]  # slice off the prompt
            text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            preds_local.append(text)
            inp_tok_local.append(int(L))
            out_tok_local.append(int(new_tokens.shape[0]))
        return preds_local, inp_tok_local, out_tok_local

    preds, inp_tok, out_tok = _decode_batch()

    stats = [
        build_generation_stats(
            timer=timer,
            n_images=len(images_batch[i]),
            input_tokens=inp_tok[i],
            output_tokens=out_tok[i],
        )
        for i in range(bs)
    ]

    if single_mode:
        return preds[0], stats[0]
    return preds, stats
