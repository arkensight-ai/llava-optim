from __future__ import annotations
from typing import List, Tuple, Dict, Any
import argparse
import time
import torch
import pandas as pd
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

# High-quality downscale
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaVA OneVision on a CSV of prompts/images, with profiling."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help=("CSV with columns: image_paths,prompt,answer. "
              "Multiple images in image_paths separated by ';'."),
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        help="HF repo id.",
    )
    parser.add_argument(
        "--four-bit",
        action="store_true",
        help="Load model in 4-bit quantization.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=50,
        help="Max new tokens to generate.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (only used if sampling).",
    )
    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (greedy decode).",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1280,
        help="Resize each image so its longest side <= this.",
    )
    parser.add_argument(
        "--mi-mode",
        choices=["images", "frames"],
        default="images",
        help=("Multi-image mode: "
              "'images' = one <image> placeholder PER image (default); "
              "'frames' = ONE placeholder covering all images (nested-list)."),
    )
    return parser.parse_args()


def load_model(
    model_id: str,
    four_bit: bool,
) -> Tuple[LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor]:
    quantization_config = None
    if four_bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    if four_bit:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
        )

    processor = LlavaOnevisionProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def resize_to_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), RESAMPLE)


def build_prompt_for_images(
    processor: LlavaOnevisionProcessor,
    user_text: str,
    num_images: int,
    mi_mode: str,  # "images" or "frames"
) -> str:
    """
    images mode  : N {"type": "image"} entries + text
    frames mode  : 1 {"type": "image"} entry + text (we'll pass nested list later)
    """
    content_blocks = []
    if mi_mode == "images":
        for _ in range(num_images):
            content_blocks.append({"type": "image"})
    else:  # frames
        content_blocks.append({"type": "image"})
    content_blocks.append({"type": "text", "text": user_text})

    conversation = [{"role": "user", "content": content_blocks}]
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )
    return prompt


def prepare_inputs_from_csv(
    csv_path: str,
    processor: LlavaOnevisionProcessor,
    max_image_side: int,
    mi_mode: str,
) -> Tuple[List[List[Image.Image]], List[str], List[str], List[str]]:
    df = pd.read_csv(csv_path)
    images_batch: List[List[Image.Image]] = []
    model_prompts: List[str] = []
    user_prompts: List[str] = []
    answers: List[str] = []

    for _, row in df.iterrows():
        raw_paths = [p.strip() for p in str(row["image_paths"]).split(";")]
        imgs: List[Image.Image] = []
        for p in raw_paths:
            im = Image.open(p).convert("RGB")
            imgs.append(resize_to_max_side(im, max_image_side))

        question = str(row["prompt"])
        prompt_for_model = build_prompt_for_images(
            processor=processor,
            user_text=question,
            num_images=len(imgs),
            mi_mode=mi_mode,
        )
        gt_val = row.get("answer", "")
        gt_val = "" if pd.isna(gt_val) else str(gt_val)

        images_batch.append(imgs)
        model_prompts.append(prompt_for_model)
        user_prompts.append(question)
        answers.append(gt_val)

    return images_batch, model_prompts, user_prompts, answers


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
    """
    Count tiles per image using the processor's image_processor.
    No 'videos' kwarg here (it isn't supported).
    """
    tokens_per_tile = getattr(processor, "tokens_per_tile", 256)
    per_image_info: List[Dict[str, Any]] = []
    total_tiles = 0

    for idx, img in enumerate(images_for_prompt):
        single_out = processor.image_processor(images=[img])  # <- FIXED
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
    mi_mode: str,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
) -> Tuple[str | None, Dict[str, Any]]:
    # -----------------------------
    # 1) Preprocess on CPU
    # -----------------------------
    t0 = time.perf_counter()
    if mi_mode == "images":
        # ONE sample that contains multiple images; matches N <image> tokens
        raw_inputs = processor(
            images=[images_for_prompt],   # nested list
            text=[model_prompt],          # list => batch size 1
            padding=True,
            return_tensors="pt",
        )
    else:
        # frames mode: one placeholder in prompt, pass nested list
        raw_inputs = processor(
            images=[images_for_prompt],  # nested list => single visual block
            text=[model_prompt],         # list => batch size 1
            padding=True,
            return_tensors="pt",
        )
    t1 = time.perf_counter()
    prep_time_s = t1 - t0

    input_ids = raw_inputs["input_ids"]
    input_ids = raw_inputs["input_ids"]
    if mi_mode == "images":
        assert input_ids.shape[0] == 1, "images mode should produce batch size 1; pass images=[[...]] and text=[...]."
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

    # -----------------------------
    # 2) Move to device
    # -----------------------------
    device = next(model.parameters()).device
    t2 = time.perf_counter()
    inputs = _move_inputs_to_device_half_if_cuda(raw_inputs, device)
    t3 = time.perf_counter()
    move_time_s = t3 - t2

    # -----------------------------
    # 3) Generate
    # -----------------------------
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
            g0, g1 = time.perf_counter(), None
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
        "mi_mode": mi_mode,
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


def pretty_print_stats(stats: Dict[str, Any]) -> None:
    print("  --- Profiling ---")
    print(f"  mi_mode: {stats['mi_mode']}")
    print(f"  num_images: {stats['num_images']}")

    print("  per_image_tiles:")
    for info in stats["per_image_tiles"]:
        print(
            f"    img[{info['image_index']}] size={info['orig_size_wh']} "
            f"tiles={info['tiles']} vision_tokens~{info['vision_tokens']}"
        )

    print(
        f"  expected_total_tiles_individual: {stats['expected_total_tiles_individual']} "
        f"(~{stats['expected_total_vision_tokens_individual']} vision tokens)"
    )
    print(
        f"  actual pixel_values_shape: {stats['pixel_values_shape']} "
        f"-> tiles_in_batch={stats['actual_total_tiles_in_batch']} "
        f"(~{stats['actual_total_vision_tokens_in_batch']} vision tokens total)"
    )
    print(f"  tokens_per_tile (vision): {stats['tokens_per_tile']}")
    print(
        f"  pre_expansion_text_tokens: {stats['pre_expansion_text_tokens']}  "
        f"prompt_tokens_after_expansion: {stats['prompt_tokens_after_expansion']}"
    )
    print(
        f"  new_tokens_generated: {stats['new_tokens_generated']}  "
        f"tokens/sec: {stats['tokens_per_second']}"
    )
    print(
        f"  prep_time_s: {stats['prep_time_s']:.4f}  "
        f"move_to_device_time_s: {stats['move_to_device_time_s']:.4f}  "
        f"generate_time_s: {stats['generate_time_s'] if stats['generate_time_s'] is not None else 'OOM'}"
    )
    print(f"  total_time_s: {stats['total_time_s']:.4f}")
    print(f"  oom: {stats['oom']}")


def main():
    args = parse_args()

    model, processor = load_model(model_id=args.model_id, four_bit=args.four_bit)

    images_batch, model_prompts, user_prompts, answers = prepare_inputs_from_csv(
        csv_path=args.csv,
        processor=processor,
        max_image_side=args.max_image_side,
        mi_mode=args.mi_mode,
    )

    for idx in range(len(model_prompts)):
        pred, stats = generate_single_with_stats(
            model=model,
            processor=processor,
            images_for_prompt=images_batch[idx],
            model_prompt=model_prompts[idx],
            mi_mode=args.mi_mode,
            max_new_tokens=args.max_new_tokens,
            do_sample=(not args.no_sample),
            top_p=args.top_p,
        )

        print(f"\n=== Sample {idx} ===")
        print(f"Q:  {user_prompts[idx]}")
        print(f"→ Model: {pred}")
        if answers[idx] != "":
            print(f"→ GT:    {answers[idx]}")
        pretty_print_stats(stats)


if __name__ == "__main__":
    main()
