from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoConfig, LlavaOnevisionProcessor

from benchmark import PhaseTimer, build_generation_stats


@dataclass
class OnnxLlavaModel:
    decoder: ort.InferenceSession
    embed: ort.InferenceSession
    vision: ort.InferenceSession
    emb_dtype: np.dtype
    image_token_id: int
    eos_id: int
    num_kv_heads: int
    head_dim: int
    decoder_input_names: List[str]
    decoder_output_names: List[str]
    embed_input_name: str
    embed_output_name: str
    vision_input_name: str
    vision_output_name: str
    empty_past: Dict[str, np.ndarray]


# ============================== Utilities ==============================

def _alias_quant(q: Optional[str]) -> Optional[str]:
    if not q:
        return None
    q = q.lower()
    return {
        "f16": "fp16",
        "float16": "fp16",
        "bf16": "fp16",
        "q8": "uint8",
        "nf4": "bnb4",
        "bnb4_nf4": "bnb4",
        "4bit": "bnb4",
    }.get(q, q)


def _int_or(val, fallback: int) -> int:
    try:
        return int(val)
    except Exception:
        return int(fallback)


def _providers() -> List[str]:
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]


def _pick(onnx_dir: str, stem: str, quant: Optional[str]) -> str:
    if quant:
        candidate = os.path.join(onnx_dir, f"{stem}_{quant}.onnx")
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(onnx_dir, f"{stem}.onnx")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Missing {stem}(.onnx) in {onnx_dir} (tried suffix={quant!r})")


def _load_sessions(onnx_dir: str, quant: Optional[str]) -> Tuple[ort.InferenceSession, ort.InferenceSession, ort.InferenceSession]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    decoder = ort.InferenceSession(_pick(onnx_dir, "decoder_model_merged", quant), sess_options=so, providers=_providers())
    embed = ort.InferenceSession(_pick(onnx_dir, "embed_tokens", quant), sess_options=so, providers=_providers())
    vision = ort.InferenceSession(_pick(onnx_dir, "vision_encoder", quant), sess_options=so, providers=_providers())
    return decoder, embed, vision


def _onnx_type_to_np(t: str):
    return np.float16 if "float16" in t else np.float32


def _kv_seed_zeros(decoder: ort.InferenceSession, num_kv_heads: int, head_dim: int, dtype) -> Dict[str, np.ndarray]:
    seed: Dict[str, np.ndarray] = {}
    for inp in decoder.get_inputs():
        name = inp.name
        if name.startswith("past_key_values.") and (name.endswith(".key") or name.endswith(".value")):
            seed[name] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=dtype)
    return seed


def _outputs_to_next_past(decoder: ort.InferenceSession, outs: List[np.ndarray]) -> Dict[str, np.ndarray]:
    in_names = {i.name for i in decoder.get_inputs()}
    out_names = [o.name for o in decoder.get_outputs()]
    past: Dict[str, np.ndarray] = {}
    for name, arr in zip(out_names[1:], outs[1:]):
        if name.startswith("present_key_values."):
            target = name.replace("present_key_values.", "past_key_values.")
        elif name.startswith("present."):
            target = name.replace("present.", "past_key_values.")
        elif name.startswith("present_"):
            target = name.replace("present_", "past_key_values.")
        else:
            target = name.replace("present", "past_key_values")
        if target in in_names:
            past[target] = arr
    return past


def _downscale_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / float(max(w, h))
    new = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new, getattr(Image, "Resampling", Image).LANCZOS)


def _normalize_generation_inputs(
    images_input: Union[List[Any], List[List[Any]]],
    prompts_input: Union[str, List[str]],
) -> Tuple[List[List[Any]], List[str], bool]:
    if isinstance(prompts_input, str):
        if not isinstance(images_input, list):
            raise TypeError("images must be a list when passing a single prompt.")
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


# ============================== Core helpers ==============================

def load_model(
    model_id: str,
    onnx_dir: str,
    quant: Optional[Dict[str, Any]] = None,
) -> Tuple[OnnxLlavaModel, LlavaOnevisionProcessor]:
    quant_name = _alias_quant((quant or {}).get("name"))
    decoder, embed, vision = _load_sessions(onnx_dir, quant_name)

    dec_inputs = {i.name: i for i in decoder.get_inputs()}
    emb_inp = dec_inputs.get("inputs_embeds")
    emb_dtype = _onnx_type_to_np(emb_inp.type) if emb_inp is not None else np.float16

    processor = LlavaOnevisionProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(model_id)
    text_cfg = getattr(cfg, "text_config", cfg)
    num_heads = _int_or(getattr(text_cfg, "num_attention_heads", None), 14)
    hidden_size = _int_or(getattr(text_cfg, "hidden_size", None), 896)
    num_kv_heads = _int_or(getattr(text_cfg, "num_key_value_heads", None), num_heads)
    head_dim = hidden_size // max(1, num_heads)
    image_token_id = _int_or(getattr(cfg, "image_token_index", None), 151646)
    eos_id = _int_or(getattr(text_cfg, "eos_token_id", None), processor.tokenizer.eos_token_id)

    model = OnnxLlavaModel(
        decoder=decoder,
        embed=embed,
        vision=vision,
        emb_dtype=emb_dtype,
        image_token_id=image_token_id,
        eos_id=eos_id,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        decoder_input_names=[i.name for i in decoder.get_inputs()],
        decoder_output_names=[o.name for o in decoder.get_outputs()],
        embed_input_name=embed.get_inputs()[0].name,
        embed_output_name=embed.get_outputs()[0].name,
        vision_input_name=vision.get_inputs()[0].name,
        vision_output_name=vision.get_outputs()[0].name,
        empty_past=_kv_seed_zeros(decoder, num_kv_heads, head_dim, emb_dtype),
    )
    return model, processor


def _prepare_pixel_values(processor: LlavaOnevisionProcessor, images: List[Any]) -> np.ndarray:
    batch = processor.image_processor(images=images, return_tensors="np")
    pixel_values = batch["pixel_values"]
    if pixel_values.ndim == 5:
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.reshape(b * n, c, h, w)
    elif pixel_values.ndim != 4:
        raise ValueError(f"Unexpected pixel_values rank {pixel_values.ndim}; expected 4 or 5.")
    return pixel_values


def _generate_single(
    model: OnnxLlavaModel,
    processor: LlavaOnevisionProcessor,
    images: List[Any],
    prompt: str,
    max_new_tokens: int,
    timer: PhaseTimer,
) -> Tuple[str, int, int]:
    with timer.phase("tokenize"):
        tokenized = processor.tokenizer(prompt, return_tensors="np")
        input_ids = tokenized["input_ids"]

    with timer.phase("vision"):
        pixel_values = _prepare_pixel_values(processor, images)
        feats = model.vision.run([model.vision_output_name], {model.vision_input_name: pixel_values})[0]
        if feats.ndim == 3:
            feats = feats.reshape(1, feats.shape[0] * feats.shape[1], feats.shape[2])
        elif feats.ndim == 2:
            feats = feats[None, ...]
        else:
            raise ValueError(f"Unexpected feats rank {feats.ndim}; expected 2 or 3.")

    with timer.phase("embed"):
        inputs_embeds = model.embed.run([model.embed_output_name], {model.embed_input_name: input_ids})[0]

    ids = input_ids[0]
    embs = inputs_embeds[0]
    img_pos = np.where(ids == model.image_token_id)[0]
    if len(img_pos) == 0:
        merged = np.concatenate([feats[0], embs], axis=0)
    else:
        i0 = int(img_pos[0])
        merged = np.concatenate([embs[:i0], feats[0], embs[i0 + 1 :]], axis=0)
    merged = merged.astype(model.emb_dtype, copy=False)[None, ...]

    prompt_len = merged.shape[1]
    attention_mask = np.ones((1, prompt_len), dtype=np.int64)
    position_ids = np.arange(prompt_len, dtype=np.int64)[None, :]

    feed = {
        "inputs_embeds": merged,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        **model.empty_past,
    }
    feed = {k: v for k, v in feed.items() if k in model.decoder_input_names}

    with timer.phase("decode"):
        outs = model.decoder.run(model.decoder_output_names, feed)
    logits = outs[0]
    past = _outputs_to_next_past(model.decoder, outs)

    generated: List[int] = []
    for step in range(max_new_tokens):
        next_id = int(logits[:, -1, :].argmax(-1)[0])
        if next_id == model.eos_id:
            break
        generated.append(next_id)

        with timer.phase("embed"):
            new_embed = model.embed.run(
                [model.embed_output_name],
                {model.embed_input_name: np.array([[next_id]], dtype=np.int64)},
            )[0]
        new_embed = new_embed.astype(model.emb_dtype, copy=False)

        step_pos = np.array([[prompt_len + step]], dtype=np.int64)
        step_att = np.ones((1, prompt_len + step + 1), dtype=np.int64)
        step_feed = {
            "inputs_embeds": new_embed,
            "attention_mask": step_att,
            "position_ids": step_pos,
            **past,
        }
        step_feed = {k: v for k, v in step_feed.items() if k in model.decoder_input_names}

        with timer.phase("decode"):
            outs = model.decoder.run(model.decoder_output_names, step_feed)
        logits = outs[0]
        past = _outputs_to_next_past(model.decoder, outs)

    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, int(prompt_len), len(generated)


@torch.inference_mode()
def generate_with_stats(
    model: OnnxLlavaModel,
    processor: LlavaOnevisionProcessor,
    images: Union[List[Any], List[List[Any]]],
    model_prompts: Union[str, List[str]],
    max_new_tokens: int = 256,
    do_sample: bool = False,
    top_p: float = 0.9,
    temperature: float = 1.0,
):
    if do_sample:
        raise ValueError("ONNX backend currently supports greedy decoding only (do_sample=False).")

    images_batch, prompts_batch, single_mode = _normalize_generation_inputs(images, model_prompts)
    if not prompts_batch:
        return ("", {}) if single_mode else ([], [])

    preds: List[str] = []
    stats: List[Dict[str, Any]] = []

    for imgs, prompt in zip(images_batch, prompts_batch):
        timer = PhaseTimer()
        text, in_tok, out_tok = _generate_single(
            model=model,
            processor=processor,
            images=imgs,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            timer=timer,
        )
        preds.append(text)
        stats.append(
            build_generation_stats(
                timer=timer,
                n_images=len(imgs),
                input_tokens=in_tok,
                output_tokens=out_tok,
            )
        )

    if single_mode:
        return preds[0], stats[0]
    return preds, stats


# ============================== CLI ==============================

def _cli_prompt(processor: LlavaOnevisionProcessor, text: str, n_images: int) -> str:
    blocks = [{"type": "image"} for _ in range(n_images)]
    blocks.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": blocks}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def main():
    ap = argparse.ArgumentParser("LLaVA OneVision ONNX inference (integrated backend)")
    ap.add_argument("--onnx-dir", required=True, help="Folder with ONNX files (vision, embed, decoder)")
    ap.add_argument("--model-id", default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    ap.add_argument("--images", nargs="+", default=[], help="One or more image paths")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--quant", default=None, help="Optional quant suffix: fp16, bnb4, uint8, ...")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-image-side", type=int, default=384)
    args = ap.parse_args()

    quant_cfg = {"name": args.quant} if args.quant else None
    model, processor = load_model(
        model_id=args.model_id,
        onnx_dir=args.onnx_dir,
        quant=quant_cfg,
    )

    images = [[_downscale_max_side(Image.open(p).convert("RGB"), args.max_image_side) for p in args.images]]
    prompt_text = _cli_prompt(processor, args.prompt, len(images[0]))

    pred, _ = generate_with_stats(
        model=model,
        processor=processor,
        images=images,
        model_prompts=[prompt_text],
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
    )
    print(pred)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
