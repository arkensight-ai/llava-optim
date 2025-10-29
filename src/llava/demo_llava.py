from __future__ import annotations
from typing import List, Tuple
import argparse
import torch
import pandas as pd
from PIL import Image

from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)

# Pillow compatibility: Image.Resampling.LANCZOS is the new name
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLaVA OneVision on a CSV of prompts/images."
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help=(
            "Path to CSV with columns: image_paths,prompt,answer. "
            "Multiple images in image_paths are separated by ';'."
        ),
    )

    parser.add_argument(
        "--model-id",
        type=str,
        default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        help="HF model repo id.",
    )

    parser.add_argument(
        "--four-bit",
        action="store_true",
        help="Load model in 4-bit quantization to save VRAM.",
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
        help="Top-p sampling cutoff (used only if sampling).",
    )

    parser.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (use greedy decoding).",
    )

    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1280,
        help=(
            "Resize each image so its longest side is at most this many pixels. "
            "Lower this if you still OOM on multi-image."
        ),
    )

    return parser.parse_args()


def load_model(
    model_id: str,
    four_bit: bool,
) -> Tuple[LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor]:
    """
    Load the Llava OneVision model + processor.

    We'll:
    - optionally quantize to 4-bit
    - force left padding on tokenizer (recommended for batched generation). :contentReference[oaicite:4]{index=4}
    """

    quantization_config = None
    if four_bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    if four_bit:
        # bitsandbytes path: don't also pass dtype
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        # half precision path (uses new `dtype=` kwarg in recent transformers)
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
    """
    Downscale 'img' so that max(width, height) <= max_side.
    Keeps aspect ratio. Uses high-quality downsampling.

    This is important because OneVision patchifies high-res images into lots
    of visual tokens, which eats VRAM. Resizing keeps you under OOM. :contentReference[oaicite:5]{index=5}
    """
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img  # already small enough

    scale = max_side / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), RESAMPLE)


def build_prompt_for_images(
    processor: LlavaOnevisionProcessor,
    user_text: str,
    num_images: int,
) -> str:
    """
    Build a *single-turn* conversation for ONE CSV row.

    Critical bits:
    - We include ONE {"type": "image"} *per image* we plan to feed.
      This ensures the chat template inserts the correct number of
      <image> placeholder tokens, and the model later verifies that
      the # of <image> tokens == # of images' visual features. :contentReference[oaicite:6]{index=6}

    - We put the images first in the "content" list, then the user's text,
      which matches the official examples. :contentReference[oaicite:7]{index=7}
    """

    content_blocks = []
    for _ in range(num_images):
        content_blocks.append({"type": "image"})
    content_blocks.append({"type": "text", "text": user_text})

    conversation = [
        {
            "role": "user",
            "content": content_blocks,
        },
    ]

    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )

    return prompt


def prepare_inputs_from_csv(
    csv_path: str,
    processor: LlavaOnevisionProcessor,
    max_image_side: int,
) -> Tuple[List[List[Image.Image]], List[str], List[str], List[str]]:
    """
    Read a CSV with columns:
        image_paths,prompt,answer
    where image_paths can contain multiple paths separated by ';'.

    Returns parallel lists length N:
        images_batch[i]  : list[Image] for row i, resized
        model_prompts[i] : string prompt specialized for the model
        user_prompts[i]  : raw question from CSV
        answers[i]       : ground truth answer (optional)
    """
    df = pd.read_csv(csv_path)

    images_batch: List[List[Image.Image]] = []
    model_prompts: List[str] = []
    user_prompts: List[str] = []
    answers: List[str] = []

    for _, row in df.iterrows():
        # 1. Load & resize all images for this row
        raw_paths = [p.strip() for p in str(row["image_paths"]).split(";")]
        imgs: List[Image.Image] = []
        for p in raw_paths:
            im = Image.open(p).convert("RGB")
            im = resize_to_max_side(im, max_image_side)
            imgs.append(im)

        # 2. Natural language question
        question = str(row["prompt"])

        # 3. Build the model-facing prompt
        #    IMPORTANT: num_images=len(imgs) so we emit that many {"type":"image"} tokens
        prompt_for_model = build_prompt_for_images(
            processor=processor,
            user_text=question,
            num_images=len(imgs),
        )

        # 4. Ground truth answer if present
        gt_val = row.get("answer", "")
        gt_val = "" if pd.isna(gt_val) else str(gt_val)

        # Collect
        images_batch.append(imgs)
        model_prompts.append(prompt_for_model)
        user_prompts.append(question)
        answers.append(gt_val)

    return images_batch, model_prompts, user_prompts, answers


def _move_inputs_to_device_half_if_cuda(
    inputs: dict,
    device: torch.device,
) -> dict:
    """
    Take processor(...) output dict and:
      - move tensors to model.device
      - if CUDA, cast float32 vision tensors down to float16 to save VRAM
        (pixel_values etc.)
    """
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if device.type == "cuda" and v.dtype == torch.float32:
                v = v.to(dtype=torch.float16)
        out[k] = v
    return out


def _strip_prefix(decoded: str) -> str:
    """
    After decoding we often get:
      "user ... assistant\n<answer>"
    We'll try to return just the assistant answer.
    """
    token = "assistant"
    if token in decoded:
        decoded = decoded.split(token, 1)[1]
    return decoded.strip()


def generate_single(
    model: LlavaOnevisionForConditionalGeneration,
    processor: LlavaOnevisionProcessor,
    images_for_prompt: List[Image.Image],
    model_prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
) -> str:
    """
    Generate one answer for one row.

    images_for_prompt: [img1, img2, ...] for THIS row
    model_prompt: string built by build_prompt_for_images() for THIS row

    The key detail here:
    - We wrap `images_for_prompt` in an outer list, and the prompt string
      in a list too.
      images = [[img1, img2, ...]]
      text    = [prompt]
    This tells the processor "batch size = 1, but that 1 sample has several images".
    This matches OneVision's multi-image design and avoids the 'index by index'
    broadcast bug you hit. :contentReference[oaicite:8]{index=8}
    """

    raw_inputs = processor(
        images=[images_for_prompt],  # nested list -> single sample with multiple images
        text=[model_prompt],         # list of length 1 -> batch size 1
        padding=True,
        return_tensors="pt",
    )

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["top_p"] = top_p

    device = next(model.parameters()).device
    inputs = _move_inputs_to_device_half_if_cuda(raw_inputs, device)

    output_tokens = model.generate(**inputs, **gen_kwargs)

    decoded_full = processor.batch_decode(
        output_tokens,
        skip_special_tokens=True,
    )[0]

    return _strip_prefix(decoded_full)


def main():
    args = parse_args()

    # 1. Load model + processor
    model, processor = load_model(
        model_id=args.model_id,
        four_bit=args.four_bit,
    )

    # 2. Build per-row inputs
    (
        images_batch,
        model_prompts,
        user_prompts,
        answers,
    ) = prepare_inputs_from_csv(
        csv_path=args.csv,
        processor=processor,
        max_image_side=args.max_image_side,
    )

    # 3. Generate row by row
    for idx in range(len(model_prompts)):
        pred = generate_single(
            model=model,
            processor=processor,
            images_for_prompt=images_batch[idx],
            model_prompt=model_prompts[idx],
            max_new_tokens=args.max_new_tokens,
            do_sample=(not args.no_sample),
            top_p=args.top_p,
        )

        # 4. Pretty-print
        print(f"\n=== Sample {idx} ===")
        print(f"Q:  {user_prompts[idx]}")
        print(f"→ Model: {pred}")
        if answers[idx] != "":
            print(f"→ GT:    {answers[idx]}")


if __name__ == "__main__":
    main()
