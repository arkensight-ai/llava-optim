from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional, Union
import torch
from benchmark import PhaseTimer, build_generation_stats
from transformers import BitsAndBytesConfig, AutoModelForVision2Seq, AutoProcessor

def _maybe_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name: return None
    name = str(name).lower()
    if name in ("fp16", "float16", "torch.float16"): return torch.float16
    if name in ("bf16", "bfloat16", "torch.bfloat16"): return torch.bfloat16
    return torch.float32

def _to_vllm_dtype(label: Any) -> str:
    if isinstance(label, torch.dtype):
        if label == torch.bfloat16: return "bfloat16"
        return "float16"
    s = str(label or "fp16").lower()
    return "bfloat16" if "bf16" in s else "float16"

def load_model(model_id: str, quant: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any]:
    if quant is None: quant = {"name": "fp16", "dtype": "fp16"}
    from_kwargs = dict(device_map="auto", trust_remote_code=True)
    name = str(quant.get("name", "fp16")).lower()
    
    if name.startswith("bnb4"):
        from_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    else:
        from_kwargs["torch_dtype"] = _maybe_dtype(quant.get("dtype"))

    model = AutoModelForVision2Seq.from_pretrained(model_id, **from_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(processor.tokenizer, "padding_side"): processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor

def load_model_vllm(model_id: str, quant: Optional[Dict[str, Any]] = None, 
                    max_model_len: int = 8192, gpu_memory_utilization: float = 0.9) -> Tuple[Any, Any]:
    from vllm import LLM
    llm = LLM(model=model_id, trust_remote_code=True, dtype=_to_vllm_dtype(quant), 
              max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
              limit_mm_per_prompt={"image": 4})
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return llm, processor

def _normalize_generation_inputs(images_input, prompts_input):
    if isinstance(prompts_input, str): return [images_input], [prompts_input], True
    return images_input, prompts_input, False

@torch.inference_mode()
def generate_with_stats(model, processor, images, model_prompts, **kwargs):
    images_batch, prompts_batch, _ = _normalize_generation_inputs(images, model_prompts)
    bs = len(prompts_batch)
    timer = PhaseTimer()
    
    with timer.phase("inference"):
        enc = processor(text=prompts_batch, images=images_batch, padding=True, return_tensors="pt").to(model.device)
        torch.cuda.synchronize()
        with timer.phase("generate"):
            gen_out = model.generate(**enc, use_cache=True, max_new_tokens=kwargs.get("max_new_tokens", 10), do_sample=False)
            torch.cuda.synchronize()

    amortized_total = timer.total() / bs
    preds, out_tok, inp_tok = [], [], []
    for i, L in enumerate((enc["input_ids"] != processor.tokenizer.pad_token_id).sum(dim=1).tolist()):
        text = processor.tokenizer.decode(gen_out[i, L:], skip_special_tokens=True).strip()
        preds.append(text)
        inp_tok.append(len(processor.tokenizer(prompts_batch[i], add_special_tokens=False)["input_ids"]))
        out_tok.append(len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]))

    stats = []
    for i in range(bs):
        s = build_generation_stats(timer, len(images_batch[i]), inp_tok[i], out_tok[i])
        s.update({"t_total_s": amortized_total, "tokens_per_s": out_tok[i] / amortized_total if amortized_total > 0 else 0.0})
        stats.append(s)
    return preds, stats

def generate_with_stats_vllm(model, processor, images, model_prompts, **kwargs):
    from vllm import SamplingParams
    images_batch, prompts_batch, _ = _normalize_generation_inputs(images, model_prompts)
    bs = len(prompts_batch)
    timer = PhaseTimer()
    
    with timer.phase("inference"):
        vllm_inputs = [{"prompt": p, "multi_modal_data": {"image": imgs if len(imgs) > 1 else imgs[0]}} 
                       for imgs, p in zip(images_batch, prompts_batch)]
        sampling_params = SamplingParams(max_tokens=kwargs.get("max_new_tokens", 10), temperature=0.0)
        outputs = model.generate(vllm_inputs, sampling_params=sampling_params)

    amortized_total = timer.total() / bs
    preds, out_tok, inp_tok = [], [], []
    for i, out in enumerate(outputs):
        text = out.outputs[0].text.strip()
        preds.append(text)
        out_tok.append(len(processor.tokenizer(text, add_special_tokens=False)["input_ids"]))
        inp_tok.append(len(processor.tokenizer(prompts_batch[i], add_special_tokens=False)["input_ids"]))

    stats = []
    for i in range(bs):
        s = build_generation_stats(timer, len(images_batch[i]), inp_tok[i], out_tok[i])
        s.update({"t_total_s": amortized_total, "tokens_per_s": out_tok[i] / amortized_total if amortized_total > 0 else 0.0})
        stats.append(s)
    return preds, stats