from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Union

import torch
from PIL import Image
from benchmark import PhaseTimer, build_generation_stats
from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


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


def _to_vllm_dtype(label: Any) -> str:
    """
    Map various config labels ('fp16', torch.float16, etc.) to
    vLLM-accepted dtype strings: 'auto','half','float16','bfloat16','float','float32'.
    """
    if isinstance(label, torch.dtype):
        if label == torch.float16:
            return "float16"
        if label == torch.bfloat16:
            return "bfloat16"
        if label == torch.float32:
            return "float32"
        # Fallback
        return "float16"

    if label is None:
        return "float16"

    s = str(label).lower()
    if s in ("fp16", "float16", "half", "torch.float16"):
        return "float16"
    if s in ("bf16", "bfloat16", "torch.bfloat16"):
        return "bfloat16"
    if s in ("fp32", "float32", "float", "torch.float32"):
        return "float32"
    if s == "auto":
        return "auto"

    # Unknown label: be conservative
    return "float16"


# ---------------------------------------------------------------------
# HF model loader
# ---------------------------------------------------------------------


def load_model(
    model_id: str,
    quant: Optional[Dict[str, Any]] = None,
) -> Tuple[LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor]:
    """
    Load LLaVA-OneVision via Hugging Face transformers, with optional 4/8-bit quantization.
    """

    if quant is None:
        print("[Cfg] No quantization specified, defaulting to fp16.")
        quant = {}
        quant.update(dict(name="fp16", dtype="fp16"))

    name = str(quant.get("name", "fp16")).lower()
    dtype = _maybe_dtype(quant.get("dtype"))

    # Build kwargs for from_pretrained
    from_kwargs: Dict[str, Any] = dict(device_map="auto")

    if name.startswith("bnb4") or quant.get("load_in_4bit", False):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=_maybe_dtype(
                quant.get("bnb_4bit_compute_dtype")
            )
            or torch.bfloat16,
            bnb_4bit_use_double_quant=bool(
                quant.get("bnb_4bit_use_double_quant", True)
            ),
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

    print(f"[HF] Loading {model_id} with config: {from_kwargs}")

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        **from_kwargs,
    )
    processor = LlavaOnevisionProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


# ---------------------------------------------------------------------
# vLLM model loader
# ---------------------------------------------------------------------


def load_model_vllm(
    model_id: str,
    quant: Optional[Dict[str, Any]] = None,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
) -> Tuple[Any, LlavaOnevisionProcessor]:
    """
    Load LLaVA-OneVision via vLLM.LLM. We still use the HF processor for
    chat templates and tokenization so that stats are consistent across backends.
    """
    try:
        from vllm import LLM
    except Exception as e:
        raise RuntimeError(
            "Failed to import vllm. This is usually a binary compatibility "
            "issue between vLLM, PyTorch and CUDA (or multiple torch installs). "
            "Install vLLM via `uv add vllm` and ensure torch/vllm match.\n\n"
            f"Original error:\n{type(e).__name__}: {e}"
        ) from e

    # Derive dtype from quant config if given
    raw_label = None
    if quant is not None:
        raw_label = (
            quant.get("dtype")
            or quant.get("torch_dtype")
            or quant.get("name")
        )
    dtype = _to_vllm_dtype(raw_label)

    print(
        f"[vLLM] Loading {model_id} with dtype={dtype}, "
        f"max_model_len={max_model_len}, gpu_memory_utilization={gpu_memory_utilization}."
    )

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        dtype=dtype,  # must be 'float16','bfloat16', etc. – NOT 'fp16'
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 4},
    )

    processor = LlavaOnevisionProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"

    return llm, processor


# ---------------------------------------------------------------------
# Optional helper: inspect image tiling (HF processor)
# ---------------------------------------------------------------------


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


# ---------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------


def _token_lengths(input_ids: torch.Tensor, pad_id: int) -> List[int]:
    # counts non-pad tokens per row
    return (input_ids != pad_id).sum(dim=1).tolist()


def _normalize_generation_inputs(
    images_input: Union[List[Any], List[List[Any]]],
    prompts_input: Union[str, List[str]],
) -> Tuple[List[List[Any]], List[str], bool]:
    """
    Normalize inputs into:
      - images_batch: List[List[image]]
      - prompts_batch: List[str]
      - single_mode: bool
    """
    if isinstance(prompts_input, str):
        if not isinstance(images_input, list):
            raise TypeError(
                "images must be a list of images when passing a single prompt."
            )
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
        raise TypeError(
            "For batched generation, images must be a list of image lists."
        )
    n_imgs = len(images_input[0])
    if not all(len(sample) == n_imgs for sample in images_input):
        raise ValueError(
            "For simple batching, all samples must have the same number of images."
        )
    return images_input, prompts_input, False


# ---------------------------------------------------------------------
# HF backend generation
# ---------------------------------------------------------------------


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
):
    """
    Unified generation helper for the HF backend.
    Pass a single prompt (str) with a list of images for single-mode,
    or lists of prompts/images for batch-mode.

    Token counts are computed by re-tokenizing the text with the same
    tokenizer as the vLLM backend, to keep stats comparable.
    """
    images_batch, prompts_batch, single_mode = _normalize_generation_inputs(
        images, model_prompts
    )
    bs = len(prompts_batch)
    if bs == 0:
        return ("", {}) if single_mode else ([], [])

    timer = PhaseTimer()

    @timer.measure("encode")
    def _encode_batch():
        enc_local = processor(
            text=prompts_batch,
            images=images_batch,  # list[list[PIL.Image]]
            padding=True,  # pad text to max length in the batch
            return_tensors="pt",
        )
        device = getattr(
            model,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        return enc_local.to(device)

    enc = _encode_batch()

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

    @timer.measure("decode")
    def _decode_batch():
        # Use HF's encoded prompt lengths only to slice off the prompt;
        # for stats we re-tokenize text explicitly.
        pad_id = processor.tokenizer.pad_token_id
        input_lens = _token_lengths(enc["input_ids"], pad_id=pad_id)
        preds_local: list[str] = []
        inp_tok_local: list[int] = []
        out_tok_local: list[int] = []

        for i, L in enumerate(input_lens):
            new_tokens = gen_out[i, L:]  # slice off the prompt
            text = processor.tokenizer.decode(
                new_tokens, skip_special_tokens=True
            ).strip()
            preds_local.append(text)

            # Re-tokenize for consistent token counts across backends
            enc_prompt = processor.tokenizer(
                prompts_batch[i],
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            inp_tok_local.append(len(enc_prompt["input_ids"]))

            enc_out = processor.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            out_tok_local.append(len(enc_out["input_ids"]))

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


# ---------------------------------------------------------------------
# vLLM backend generation
# ---------------------------------------------------------------------


def generate_with_stats_vllm(
    model,
    processor,
    images: Union[List[Any], List[List["PIL.Image.Image"]]],
    model_prompts: Union[str, List[str]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    """
    vLLM-based generation mirroring generate_with_stats, but using vLLM.LLM.
    We re-tokenize prompts and outputs with the same HF tokenizer so that
    token stats are directly comparable to the HF backend.
    """
    try:
        from vllm import SamplingParams
    except Exception as e:
        raise RuntimeError(
            "Failed to import vllm in generate_with_stats_vllm. "
            "This is usually a binary compatibility issue between vLLM, PyTorch and CUDA.\n\n"
            f"Original error:\n{type(e).__name__}: {e}"
        ) from e

    images_batch, prompts_batch, single_mode = _normalize_generation_inputs(
        images, model_prompts
    )
    bs = len(prompts_batch)
    if bs == 0:
        return ("", {}) if single_mode else ([], [])

    timer = PhaseTimer()

    @timer.measure("encode")
    def _encode_batch():
        vllm_inputs: List[Dict[str, Any]] = []
        inp_tok_local: List[int] = []

        for imgs, prompt in zip(images_batch, prompts_batch):
            mm = imgs if len(imgs) > 1 else imgs[0]
            vllm_inputs.append(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": mm},
                }
            )

            # Re-tokenize prompt to count input tokens
            enc_prompt = processor.tokenizer(
                prompt,
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            inp_tok_local.append(len(enc_prompt["input_ids"]))

        return vllm_inputs, inp_tok_local

    vllm_inputs, inp_tok = _encode_batch()

    @timer.measure("generate")
    def _generate_batch():
        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature if do_sample else 0.0,
            top_p=top_p,
        )
        return model.generate(vllm_inputs, sampling_params=sampling_params)

    outputs = _generate_batch()

    @timer.measure("decode")
    def _decode_batch():
        preds_local: List[str] = []
        out_tok_local: List[int] = []

        for out in outputs:
            text = out.outputs[0].text.strip()
            preds_local.append(text)

            enc_out = processor.tokenizer(
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_tensors=None,
            )
            out_tok_local.append(len(enc_out["input_ids"]))

        return preds_local, out_tok_local

    preds, out_tok = _decode_batch()

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
