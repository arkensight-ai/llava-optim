from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional, Union

import torch
from PIL import Image
from benchmark import PhaseTimer, build_generation_stats
from transformers import (
    BitsAndBytesConfig,
    AutoModelForVision2Seq,  # Updated from LlavaOnevision
    AutoProcessor,           # Updated from LlavaOnevision
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
    if isinstance(label, torch.dtype):
        if label == torch.float16: return "float16"
        if label == torch.bfloat16: return "bfloat16"
        if label == torch.float32: return "float32"
        return "float16"

    if label is None: return "float16"

    s = str(label).lower()
    if s in ("fp16", "float16", "half", "torch.float16"): return "float16"
    if s in ("bf16", "bfloat16", "torch.bfloat16"): return "bfloat16"
    if s in ("fp32", "float32", "float", "torch.float32"): return "float32"
    if s == "auto": return "auto"
    return "float16"


# ---------------------------------------------------------------------
# Model Loaders
# ---------------------------------------------------------------------

def load_model(
    model_id: str,
    quant: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Load any Vision-Language Model via HF Auto classes."""

    if quant is None:
        print("[Cfg] No quantization specified, defaulting to fp16.")
        quant = {"name": "fp16", "dtype": "fp16"}

    name = str(quant.get("name", "fp16")).lower()
    dtype = _maybe_dtype(quant.get("dtype"))

    from_kwargs: Dict[str, Any] = dict(device_map="auto", trust_remote_code=True)

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
        if dtype is None:
            dtype = torch.float16
        from_kwargs["torch_dtype"] = dtype

    print(f"[HF] Loading {model_id} with config: {from_kwargs}")

    model = AutoModelForVision2Seq.from_pretrained(model_id, **from_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    if hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"
        
    model.eval()
    return model, processor


def load_model_vllm(
    model_id: str,
    quant: Optional[Dict[str, Any]] = None,
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
) -> Tuple[Any, Any]:
    """Load VLM via vLLM with AutoProcessor for tokenization parity."""
    try:
        from vllm import LLM
    except Exception as e:
        raise RuntimeError(f"Failed to import vllm: {e}") from e

    raw_label = None
    if quant is not None:
        raw_label = quant.get("dtype") or quant.get("torch_dtype") or quant.get("name")
    dtype = _to_vllm_dtype(raw_label)

    print(f"[vLLM] Loading {model_id} with dtype={dtype}, util={gpu_memory_utilization}")

    llm = LLM(
        model=model_id,
        trust_remote_code=True,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 4},
    )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor.tokenizer, "padding_side"):
        processor.tokenizer.padding_side = "left"

    return llm, processor


# ---------------------------------------------------------------------
# Generation Helpers
# ---------------------------------------------------------------------

def _token_lengths(input_ids: torch.Tensor, pad_id: int) -> List[int]:
    return (input_ids != pad_id).sum(dim=1).tolist()


def _normalize_generation_inputs(
    images_input: Union[List[Any], List[List[Any]]],
    prompts_input: Union[str, List[str]],
) -> Tuple[List[List[Any]], List[str], bool]:
    if isinstance(prompts_input, str):
        return [images_input], [prompts_input], True
    return images_input, prompts_input, False


@torch.inference_mode()
def generate_with_stats(
    model, processor, images, model_prompts,
    max_new_tokens: int = 256, do_sample: bool = False,
    top_p: float = 0.9, temperature: float = 1.0,
):
    images_batch, prompts_batch, single_mode = _normalize_generation_inputs(images, model_prompts)
    bs = len(prompts_batch)
    if bs == 0: return ("", {}) if single_mode else ([], [])

    timer = PhaseTimer()

    with timer.phase("encode"):
        enc = processor(text=prompts_batch, images=images_batch, padding=True, return_tensors="pt")
        enc = enc.to(model.device)

    with timer.phase("generate"):
        torch.cuda.synchronize()
        gen_out = model.generate(
            **enc, use_cache=True, max_new_tokens=max_new_tokens,
            do_sample=do_sample, top_p=top_p, temperature=temperature,
        )
        torch.cuda.synchronize()

    with timer.phase("decode"):
        pad_id = processor.tokenizer.pad_token_id
        input_lens = _token_lengths(enc["input_ids"], pad_id=pad_id)
        preds, inp_tok, out_tok = [], [], []

        for i, L in enumerate(input_lens):
            new_tokens = gen_out[i, L:]
            text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            preds.append(text)

            # Re-tokenize for consistency
            inp_tok.append(len(processor.tokenizer(prompts_batch[i], add_special_tokens=False)["input_ids"]))
            out_tok.append(len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]))

    stats = [build_generation_stats(timer, len(images_batch[i]), inp_tok[i], out_tok[i]) for i in range(bs)]

    return (preds[0], stats[0]) if single_mode else (preds, stats)


def generate_with_stats_vllm(
    model, processor, images, model_prompts,
    max_new_tokens: int = 256, do_sample: bool = False,
    top_p: float = 0.9, temperature: float = 1.0,
):
    from vllm import SamplingParams
    images_batch, prompts_batch, single_mode = _normalize_generation_inputs(images, model_prompts)
    bs = len(prompts_batch)
    if bs == 0: return ("", {}) if single_mode else ([], [])

    timer = PhaseTimer()

    with timer.phase("encode"):
        vllm_inputs, inp_tok = [], []
        for imgs, prompt in zip(images_batch, prompts_batch):
            vllm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": imgs if len(imgs) > 1 else imgs[0]}})
            inp_tok.append(len(processor.tokenizer(prompt, add_special_tokens=False)["input_ids"]))

    with timer.phase("generate"):
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature if do_sample else 0.0, top_p=top_p)
        outputs = model.generate(vllm_inputs, sampling_params=sampling_params)

    with timer.phase("decode"):
        preds, out_tok = [], []
        for out in outputs:
            text = out.outputs[0].text.strip()
            preds.append(text)
            out_tok.append(len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]))

    stats = [build_generation_stats(timer, len(images_batch[i]), inp_tok[i], out_tok[i]) for i in range(bs)]

    return (preds[0], stats[0]) if single_mode else (preds, stats)