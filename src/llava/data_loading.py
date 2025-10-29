
from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from PIL import Image
from transformers import LlavaOnevisionProcessor

# High-quality downscale
RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS

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
) -> str:
    """
    Build a *single-turn* conversation with one <image> placeholder per image.
    """
    content_blocks = []
    for _ in range(num_images):
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
) -> Tuple[List[List[Image.Image]], List[str], List[str], List[str]]:
    """Read CSV rows and return images, prompts, user texts, answers."""
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

        question = str(row["prompt"])  # user text
        prompt_for_model = build_prompt_for_images(
            processor=processor,
            user_text=question,
            num_images=len(imgs),
        )

        gt_val = row.get("answer", "")
        gt_val = "" if pd.isna(gt_val) else str(gt_val)

        images_batch.append(imgs)
        model_prompts.append(prompt_for_model)
        user_prompts.append(question)
        answers.append(gt_val)

    return images_batch, model_prompts, user_prompts, answers
