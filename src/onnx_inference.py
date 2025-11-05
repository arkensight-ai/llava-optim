# src/onnx_inference.py
# Minimal, VRAM-safe ONNX Runtime inference for LLaVA OneVision (Qwen2-0.5B-OV)
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from transformers import AutoConfig, LlavaOnevisionProcessor


# ============================== Utilities ==============================

def _alias_quant(q: str | None) -> str | None:
    """Map user quant names to filename suffixes in the ONNX folder."""
    if not q:
        return None
    q = q.lower()
    return {
        "f16": "fp16", "float16": "fp16", "bf16": "fp16",
        "q8": "uint8",
        "nf4": "bnb4", "bnb4_nf4": "bnb4", "4bit": "bnb4",
    }.get(q, q)


def _int_or(val, fallback: int) -> int:
    """Robust int cast with fallback (handles None)."""
    try:
        return int(val)
    except Exception:
        return int(fallback)


def _providers() -> List[str]:
    """Pick ORT providers; prefer CUDA if available."""
    return ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]


def _pick(onnx_dir: str, stem: str, quant: str | None) -> str:
    """Choose ONNX file by stem and optional quant suffix."""
    if quant:
        p = os.path.join(onnx_dir, f"{stem}_{quant}.onnx")
        if os.path.exists(p):
            return p
    p = os.path.join(onnx_dir, f"{stem}.onnx")
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"Missing {stem}(.onnx) in {onnx_dir} (tried suffix={quant!r})")


def _load_sessions(onnx_dir: str, quant: str | None) -> Tuple[ort.InferenceSession, ort.InferenceSession, ort.InferenceSession]:
    """Open decoder, embedding, and vision sessions."""
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    dec = ort.InferenceSession(_pick(onnx_dir, "decoder_model_merged", quant), sess_options=so, providers=_providers())
    emb = ort.InferenceSession(_pick(onnx_dir, "embed_tokens", quant), sess_options=so, providers=_providers())
    vis = ort.InferenceSession(_pick(onnx_dir, "vision_encoder", quant), sess_options=so, providers=_providers())
    return dec, emb, vis


def _onnx_type_to_np(t: str):
    """Map ONNX input type string to NumPy dtype for embeddings/KV (float16 or float32)."""
    return np.float16 if "float16" in t else np.float32


def _kv_seed_zeros(decoder_sess: ort.InferenceSession, num_kv_heads: int, head_dim: int, dtype) -> Dict[str, np.ndarray]:
    """Create zero-length KV cache tensors for all required inputs on the first call."""
    seed: Dict[str, np.ndarray] = {}
    for inp in decoder_sess.get_inputs():
        name = inp.name
        if name.startswith("past_key_values.") and (name.endswith(".key") or name.endswith(".value")):
            seed[name] = np.zeros((1, num_kv_heads, 0, head_dim), dtype=dtype)
    return seed


def _outputs_to_next_past(decoder_sess: ort.InferenceSession, outs: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Map decoder 'present.*' outputs to 'past_key_values.*' input names expected next step."""
    in_names = {i.name for i in decoder_sess.get_inputs()}
    out_names = [o.name for o in decoder_sess.get_outputs()]
    past: Dict[str, np.ndarray] = {}
    for name, arr in zip(out_names[1:], outs[1:]):  # outs[0] is logits
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
    """Resize image preserving aspect ratio so the longest side ≤ max_side."""
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    new = (int(round(w * scale)), int(round(h * scale)))
    return img.resize(new, getattr(Image, "Resampling", Image).LANCZOS)


# ============================== Core pipeline ==============================

def run_inference(
    onnx_dir: str,
    model_id: str,
    image_paths: List[str],
    prompt: str,
    quant_name: str | None,
    max_new_tokens: int,
    max_image_side: int = 384,
    debug: bool = False,
) -> str:
    """Greedy ONNX inference for LLaVA OneVision (memory-robust)."""
    # Sessions & dtypes
    quant = _alias_quant(quant_name)
    decoder_sess, embed_sess, vision_sess = _load_sessions(onnx_dir, quant)

    dec_inputs = {i.name: i for i in decoder_sess.get_inputs()}
    emb_inp = dec_inputs.get("inputs_embeds")
    emb_dtype = _onnx_type_to_np(emb_inp.type) if emb_inp is not None else np.float16

    # Processor & config
    processor = LlavaOnevisionProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    cfg = AutoConfig.from_pretrained(model_id)
    tc = getattr(cfg, "text_config", cfg)
    num_layers   = _int_or(getattr(tc, "num_hidden_layers", None), 24)
    num_heads    = _int_or(getattr(tc, "num_attention_heads", None), 14)
    hidden_size  = _int_or(getattr(tc, "hidden_size", None), 896)
    num_kv_heads = _int_or(getattr(tc, "num_key_value_heads", None), num_heads)
    head_dim     = hidden_size // max(1, num_heads)
    image_token_id = _int_or(getattr(cfg, "image_token_index", None), 151646)
    eos_id = _int_or(getattr(tc, "eos_token_id", None), processor.tokenizer.eos_token_id)

    if debug:
        print(f"[debug] layers={num_layers} heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim}")
        print(f"[debug] image_token_id={image_token_id} eos_id={eos_id} pad_id={processor.tokenizer.pad_token_id}")
        print(f"[debug] decoder inputs: {[i.name for i in decoder_sess.get_inputs()]}")
        print(f"[debug] decoder outputs: {[o.name for o in decoder_sess.get_outputs()]}")
        print(f"[debug] embed inputs:   {[i.name for i in embed_sess.get_inputs()]}")
        print(f"[debug] vision inputs:  {[i.name for i in vision_sess.get_inputs()]}")
        print(f"[debug] emb_dtype: {emb_dtype}")

    # 1) Images (downscale to limit tiles for VRAM)
    imgs = [_downscale_max_side(Image.open(p).convert("RGB"), max_image_side) for p in image_paths]

    # 2) Chat prompt (one image slot)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # 3) Tokenize text
    tok = processor.tokenizer(prompt_text, return_tensors="np")
    input_ids = tok["input_ids"]            # (1, S)
    attention_mask = tok["attention_mask"]  # (1, S)

    # 4) Vision encoder: tiles → features
    #    Using image_processor gives shape [N, 3, 384, 384] (or similar if you changed settings).
    img_batch = processor.image_processor(images=imgs, return_tensors="np")
    pixel_values = img_batch["pixel_values"]
    if pixel_values.ndim == 5:  # (B, N_img, C, H, W) → (B*N_img, C, H, W)
        b, n, c, h, w = pixel_values.shape
        pixel_values = pixel_values.reshape(b * n, c, h, w)
    elif pixel_values.ndim != 4:
        raise ValueError(f"Unexpected pixel_values rank {pixel_values.ndim}; expected 4 or 5.")
    if debug:
        print("pixel_values:", pixel_values.shape)

    v_in, v_out = vision_sess.get_inputs()[0].name, vision_sess.get_outputs()[0].name
    feats = vision_sess.run([v_out], {v_in: pixel_values})[0]
    # Normalize to (1, T, H)
    if feats.ndim == 3:       # (N, T, H) → (1, N*T, H)
        feats = feats.reshape(1, feats.shape[0] * feats.shape[1], feats.shape[2])
    elif feats.ndim == 2:     # (T, H) → (1, T, H)
        feats = feats[None, ...]
    else:
        raise ValueError(f"Unexpected feats rank {feats.ndim}; expected 2 or 3.")
    if debug:
        print("feats:", feats.shape)

    # 5) Text embeddings
    e_in, e_out = embed_sess.get_inputs()[0].name, embed_sess.get_outputs()[0].name
    inputs_embeds = embed_sess.run([e_out], {e_in: input_ids})[0]  # (1, S, H)

    # 6) Splice vision feats into the first <image> position (or prepend if none)
    ids = input_ids[0]
    embs = inputs_embeds[0]
    img_pos = np.where(ids == image_token_id)[0]
    if len(img_pos) == 0:
        merged = np.concatenate([feats[0], embs], axis=0)
    else:
        i0 = int(img_pos[0])
        merged = np.concatenate([embs[:i0], feats[0], embs[i0 + 1 :]], axis=0)
    merged = merged.astype(emb_dtype, copy=False)[None, ...]  # (1, S - 1 + T, H)
    prompt_len = merged.shape[1]

    # Masks & positions for expanded sequence
    attention_mask_full = np.ones((1, prompt_len), dtype=np.int64)
    position_ids = np.arange(prompt_len, dtype=np.int64)[None, :]

    # 7) Prime decoder (seed empty KV for required inputs)
    in_names = [i.name for i in decoder_sess.get_inputs()]
    out_names = [o.name for o in decoder_sess.get_outputs()]
    pkv_zeros = _kv_seed_zeros(decoder_sess, num_kv_heads=num_kv_heads, head_dim=head_dim, dtype=emb_dtype)

    feed = {
        "inputs_embeds": merged,
        "attention_mask": attention_mask_full,
        "position_ids": position_ids,
        **pkv_zeros,
    }
    feed = {k: v for k, v in feed.items() if k in in_names}
    outs = decoder_sess.run(out_names, feed)
    logits = outs[0]
    past = _outputs_to_next_past(decoder_sess, outs)

    # 8) Greedy decode
    generated: List[int] = []
    for t in range(max_new_tokens):
        next_id = int(logits[:, -1, :].argmax(-1)[0])
        if next_id == eos_id:
            break
        generated.append(next_id)

        new_embed = embed_sess.run([e_out], {e_in: np.array([[next_id]], dtype=np.int64)})[0]
        new_embed = new_embed.astype(emb_dtype, copy=False)

        step_pos = np.array([[prompt_len + t]], dtype=np.int64)
        step_att = np.ones((1, prompt_len + t + 1), dtype=np.int64)
        step_feed = {
            "inputs_embeds": new_embed,
            "attention_mask": step_att,
            "position_ids": step_pos,
            **past,
        }
        step_feed = {k: v for k, v in step_feed.items() if k in in_names}

        outs = decoder_sess.run(out_names, step_feed)
        logits = outs[0]
        past = _outputs_to_next_past(decoder_sess, outs)

    return processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ============================== CLI ==============================

def main():
    ap = argparse.ArgumentParser("LLaVA OneVision ONNX inference (VRAM-safe, greedy)")
    ap.add_argument("--onnx-dir", required=True, help="Folder with vision_encoder*.onnx, embed_tokens*.onnx, decoder_model_merged*.onnx")
    ap.add_argument("--model-id", default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
    ap.add_argument("--images", nargs="+", default=[], help="One or more image paths")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--quant", default=None, help="Filename suffix: fp16, q4, q4f16, int8, uint8, bnb4, ...")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--max-image-side", type=int, default=384, help="Downscale longest side before tiling")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    text = run_inference(
        onnx_dir=args.onnx_dir,
        model_id=args.model_id,
        image_paths=args.images,
        prompt=args.prompt,
        quant_name=args.quant,
        max_new_tokens=args.max_new_tokens,
        max_image_side=args.max_image_side,
        debug=args.debug,
    )
    print(text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e, file=sys.stderr)
        sys.exit(1)
