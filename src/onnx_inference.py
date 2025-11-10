from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoConfig, LlavaOnevisionProcessor

from benchmark import PhaseTimer, build_generation_stats
from debug_utils import (
    DebugController, debug_seq_report, count_tiles_from_pixel_values,
    summarize_onnx_dtypes, print_onnx_io, prefill_in_chunks_decoder,
)

# ============================== Dataclass ==============================

@dataclass
class OnnxLlavaModel:
    decoder: ort.InferenceSession
    embed: ort.InferenceSession
    vision: ort.InferenceSession

    # model/graph metadata
    emb_dtype: np.dtype
    image_token_id: int
    eos_id: int
    num_kv_heads: int
    head_dim: int
    num_heads: int
    num_layers: int
    hidden_size: int

    # names
    decoder_input_names: List[str]
    decoder_output_names: List[str]
    embed_input_name: str
    embed_output_name: str
    vision_input_name: str
    vision_output_name: str

    # seed cache
    empty_past: Dict[str, np.ndarray]


# ============================== Utilities ==============================

def _alias_quant(q: Optional[str]) -> Optional[str]:
    if not q:
        return None
    q = q.lower()
    return {
        "f16": "fp16",
        "float16": "fp16",
        "bf16": "fp16",   # ONNX packs graphs per file; prefer fp16 nets
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


def _providers(default_cuda: bool = True) -> List[str]:
    if torch.cuda.is_available() and default_cuda:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _pick(onnx_dir: str, stem: str, quant: Optional[str]) -> str:
    if quant:
        candidate = os.path.join(onnx_dir, f"{stem}_{quant}.onnx")
        if os.path.exists(candidate):
            return candidate
    candidate = os.path.join(onnx_dir, f"{stem}.onnx")
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(f"Missing {stem}(.onnx) in {onnx_dir} (tried suffix={quant!r})")


def _load_sessions(
    onnx_dir: str,
    quant: Optional[str],
    dbg: Optional[DebugController],
) -> Tuple[ort.InferenceSession, ort.InferenceSession, ort.InferenceSession, str, str, str]:
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    dec_p = _pick(onnx_dir, "decoder_model_merged", quant)
    emb_p = _pick(onnx_dir, "embed_tokens", quant)
    vis_p = _pick(onnx_dir, "vision_encoder", quant)

    providers = list(dbg.onnx.providers) if (dbg and dbg.enabled) else _providers()
    decoder = ort.InferenceSession(dec_p, sess_options=so, providers=providers)
    embed   = ort.InferenceSession(emb_p, sess_options=so, providers=providers)
    vision  = ort.InferenceSession(vis_p, sess_options=so, providers=providers)

    # Optional debug
    if dbg and dbg.enabled:
        print_onnx_io(decoder, "decoder", dbg)
        print_onnx_io(embed,   "embed",   dbg)
        print_onnx_io(vision,  "vision",  dbg)
        if dbg.detail.onnx_dtypes:
            for tag, p in [("decoder", dec_p), ("embed", emb_p), ("vision", vis_p)]:
                try:
                    counts = summarize_onnx_dtypes(p)
                    print(f"[debug/ort] {tag} initializer dtypes: {counts}")
                except Exception as e:
                    print(f"[debug/ort] {tag} summarize failed: {e}")

    return decoder, embed, vision, dec_p, emb_p, vis_p


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
        tgt = _present_to_past_name(name)
        if tgt in in_names:
            past[tgt] = arr
    return past


def _present_to_past_name(out_name: str) -> str:
    if out_name.startswith("present_key_values."):
        return out_name.replace("present_key_values.", "past_key_values.")
    if out_name.startswith("present."):
        return out_name.replace("present.", "past_key_values.")
    if out_name.startswith("present_"):
        return out_name.replace("present_", "past_key_values.")
    return out_name.replace("present", "past_key_values")


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
    dbg: Optional[DebugController] = None,
) -> Tuple[OnnxLlavaModel, LlavaOnevisionProcessor]:
    dbg = dbg or DebugController(None)

    quant_name = _alias_quant((quant or {}).get("name"))
    decoder, embed, vision, dec_p, emb_p, vis_p = _load_sessions(onnx_dir, quant_name, dbg)

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
    num_layers = _int_or(getattr(text_cfg, "num_hidden_layers", None), 24)
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
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_size=hidden_size,
        decoder_input_names=[i.name for i in decoder.get_inputs()],
        decoder_output_names=[o.name for o in decoder.get_outputs()],
        embed_input_name=embed.get_inputs()[0].name,
        embed_output_name=embed.get_outputs()[0].name,
        vision_input_name=vision.get_inputs()[0].name,
        vision_output_name=vision.get_outputs()[0].name,
        empty_past=_kv_seed_zeros(decoder, num_kv_heads, head_dim, emb_dtype),
    )
    return model, processor


def _prepare_pixel_values(processor: LlavaOnevisionProcessor, images: List[Any]) -> Tuple[np.ndarray, int]:
    """
    Return (pixel_values_4d, tiles) where pixel_values_4d is (B*T, C, H, W) with B==1.
    """
    batch = processor.image_processor(images=images, return_tensors="np")
    pixel_values = batch["pixel_values"]
    tiles = 0
    if pixel_values.ndim == 5:
        # (B, T, C, H, W) → (B*T, C, H, W)
        b, n, c, h, w = pixel_values.shape
        tiles = int(n)
        pixel_values = pixel_values.reshape(b * n, c, h, w)
    elif pixel_values.ndim == 4:
        # (B*T, C, H, W) with B==1
        tiles = int(pixel_values.shape[0])
    else:
        raise ValueError(f"Unexpected pixel_values rank {pixel_values.ndim}; expected 4 or 5.")
    return pixel_values, tiles


def _prefill_chunked_or_full(
    model: OnnxLlavaModel,
    merged: np.ndarray,               # (1, L, H)
    attention_mask: np.ndarray,       # (1, L)
    position_ids: np.ndarray,         # (1, L)
    dbg: DebugController,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run prefill (decoder priming). If chunking is enabled, use chunked prefill and return:
      (last_logits, past_dict). Otherwise do one-shot prefill.
    """
    chunk_size = int(getattr(dbg.onnx, "chunk_size", 0) or 0)

    # One-shot prefill
    if chunk_size <= 0:
        feed = {
            "inputs_embeds": merged,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            **model.empty_past,
        }
        feed = {k: v for k, v in feed.items() if k in model.decoder_input_names}
        outs = model.decoder.run(model.decoder_output_names, feed)
        logits = outs[0]
        past = _outputs_to_next_past(model.decoder, outs)
        return logits, past

    # Chunked prefill via shared utility (returns pkv list). Convert to dict.
    last_logits, pkv_list, _ = prefill_in_chunks_decoder(
        model.decoder,
        inputs_embeds_merged=merged,
        attention_mask_full=attention_mask,
        position_ids_full=position_ids,
        chunk_size=chunk_size,
        dbg=dbg,
        pkv_init=model.empty_past,
    )

    # Try to map present names → past input names deterministically.
    out_names = [o.name for o in model.decoder.get_outputs()]
    # collect present-like outputs to align with pkv_list
    present_names = [n for n in out_names if re.search(r"present", n, re.I)]
    past: Dict[str, np.ndarray] = {}

    if present_names and len(present_names) == len(pkv_list):
        for oname, arr in zip(present_names, pkv_list):
            tgt = _present_to_past_name(oname)
            if tgt in model.decoder_input_names:
                past[tgt] = arr

    # Fallback: zip by input PKV order if outputs didn't expose names clearly
    if not past:
        in_past_names = [n for n in model.decoder_input_names if n.startswith("past_key_values.")]
        if len(in_past_names) == len(pkv_list):
            for nm, arr in zip(in_past_names, pkv_list):
                past[nm] = arr
        else:
            # Last resort: build dict from model.empty_past keys order
            keys = list(model.empty_past.keys())
            for nm, arr in zip(keys, pkv_list):
                past[nm] = arr

    return last_logits, past


def _generate_single(
    model: OnnxLlavaModel,
    processor: LlavaOnevisionProcessor,
    images: List[Any],
    prompt: str,
    max_new_tokens: int,
    timer: PhaseTimer,
    dbg: DebugController,
) -> Tuple[str, int, int]:
    # --------------------- Tokenize ---------------------
    with timer.phase("tokenize"):
        tokenized = processor.tokenizer(prompt, return_tensors="np")
        input_ids = tokenized["input_ids"]

    # --------------------- Vision encode ---------------------
    with timer.phase("vision"):
        pixel_values, n_tiles = _prepare_pixel_values(processor, images)
        feats = model.vision.run([model.vision_output_name], {model.vision_input_name: pixel_values})[0]
        # standardize shape to (1, T, H)
        if feats.ndim == 3:
            # already (B, T, H) or (T, H) → normalize
            if feats.shape[0] != 1:
                feats = feats.reshape(1, feats.shape[0] * feats.shape[1], feats.shape[2])
        elif feats.ndim == 2:
            feats = feats[None, ...]
        else:
            raise ValueError(f"Unexpected feats rank {feats.ndim}; expected 2 or 3.")

    # --------------------- Text embed ---------------------
    with timer.phase("embed"):
        inputs_embeds = model.embed.run([model.embed_output_name], {model.embed_input_name: input_ids})[0]

    # --------------------- Merge (<image> replacement) ---------------------
    ids = input_ids[0]
    embs = inputs_embeds[0]
    img_pos = np.where(ids == model.image_token_id)[0]
    if len(img_pos) == 0:
        merged_seq = np.concatenate([feats[0], embs], axis=0)
    else:
        i0 = int(img_pos[0])
        merged_seq = np.concatenate([embs[:i0], feats[0], embs[i0 + 1 :]], axis=0)
    merged = merged_seq.astype(model.emb_dtype, copy=False)[None, ...]  # (1, L, H)

    prompt_len = merged.shape[1]
    attention_mask = np.ones((1, prompt_len), dtype=np.int64)
    position_ids = np.arange(prompt_len, dtype=np.int64)[None, :]

    # --------------------- Debug: seq report ---------------------
    if dbg.enabled:
        vision_tokens = int(feats.shape[1])
        text_len = int(input_ids.shape[-1])
        debug_seq_report(
            dbg,
            provider="ort",
            text_len=text_len,
            vision_tokens=vision_tokens,
            tiles=int(n_tiles),
            hidden_size=model.hidden_size,
            heads=model.num_heads,
            layers=model.num_layers,
        )

    # --------------------- Prefill (chunked or full) ---------------------
    with timer.phase("decode"):
        logits, past = _prefill_chunked_or_full(
            model=model,
            merged=merged,
            attention_mask=attention_mask,
            position_ids=position_ids,
            dbg=dbg,
        )

    # --------------------- Token loop ---------------------
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
        new_embed = new_embed.astype(model.emb_dtype, copy=False)  # (1, 1, H)

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

    if dbg.enabled and dbg.detail.gen_summary:
        print(f"[debug/ort] generated_tokens={len(generated)} (prompt={prompt_len} → total={prompt_len + len(generated)})")

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
    dbg: Optional[DebugController] = None,
):
    if do_sample:
        raise ValueError("ONNX backend currently supports greedy decoding only (do_sample=False).")

    dbg = dbg or DebugController(None)

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
            dbg=dbg,
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

    # Debug controller: if launched from Hydra caller, pass DebugController(cfg) instead.
    dbg = DebugController(None)

    quant_cfg = {"name": args.quant} if args.quant else None
    model, processor = load_model(
        model_id=args.model_id,
        onnx_dir=args.onnx_dir,
        quant=quant_cfg,
        dbg=dbg,
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
        dbg=dbg,
    )
    print(pred)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("ERROR:", exc, file=sys.stderr)
        sys.exit(1)
