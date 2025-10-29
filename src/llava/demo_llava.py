from __future__ import annotations
from typing import List, Tuple
import torch
from PIL import Image
import requests
from transformers import (
    BitsAndBytesConfig,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionProcessor,
)


def load_model(
    model_id: str = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    four_bit: bool = False,
    torch_dtype: torch.dtype = torch.float16,
) -> Tuple[LlavaOnevisionForConditionalGeneration, LlavaOnevisionProcessor]:
    """
    Load the Llava OneVision model + processor.

    Params:
        model_id: HF repo id.
        four_bit: if True, load in 4-bit quantization (saves VRAM).
        torch_dtype: dtype for compute (usually float16 on GPU).

    Returns:
        (model, processor)
    """

    quantization_config = None
    if four_bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )

    processor = LlavaOnevisionProcessor.from_pretrained(model_id)
    # For generation we want left padding so batch decode doesn't get confused.
    processor.tokenizer.padding_side = "left"

    return model, processor


def prepare_images() -> List[Image.Image]:
    """
    Fetch example images (stop sign + snowman) from the web.

    Returns:
        [snowman_img, stop_sign_img]
    """
    # You could replace these with local Image.open("path.jpg") if offline.
    snowman_url = (
        "https://huggingface.co/microsoft/kosmos-2-patch14-224/"
        "resolve/main/snowman.jpg"
    )
    stop_url = "https://www.ilankelman.org/stopsigns/australia.jpg"

    snowman_img = Image.open(requests.get(snowman_url, stream=True).raw).convert("RGB")
    stop_img = Image.open(requests.get(stop_url, stream=True).raw).convert("RGB")

    return [snowman_img, stop_img]


def build_prompts(processor: LlavaOnevisionProcessor) -> List[str]:
    """
    Build chat-style prompts for each image.

    We create two short single-turn "conversations", then ask the processor
    to turn them into actual model-ready prompt strings.
    """

    conversation_image_1 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you see in this image?"},
                {"type": "image"},
            ],
        },
    ]

    conversation_image_2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is the sign?"},
                {"type": "image"},
            ],
        },
    ]

    prompt_1 = processor.apply_chat_template(
        conversation_image_1, add_generation_prompt=True
    )
    prompt_2 = processor.apply_chat_template(
        conversation_image_2, add_generation_prompt=True
    )

    return [prompt_1, prompt_2]


def generate_batch(
    model: LlavaOnevisionForConditionalGeneration,
    processor: LlavaOnevisionProcessor,
    images: List[Image.Image],
    prompts: List[str],
    max_new_tokens: int = 50,
    do_sample: bool = True,
    top_p: float = 0.9,
) -> List[str]:
    """
    Run batched generation for (image_i, prompt_i) pairs.

    images:  [img0, img1, ...]
    prompts: [prompt_for_img0, prompt_for_img1, ...]

    Returns:
        decoded model outputs as plain strings
    """

    # Build model inputs (vision tensors + text tokens) and move to model device.
    inputs = processor(
        images=images,
        text=prompts,
        padding=True,
        return_tensors="pt",
    ).to(model.device, torch.float16)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
    }

    output_tokens = model.generate(**inputs, **gen_kwargs)

    decoded = processor.batch_decode(output_tokens, skip_special_tokens=True)
    return decoded


def main():
    # 1. Load model + processor
    model, processor = load_model(
        model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        four_bit=False,  # flip to True if you want 4-bit quantization
        torch_dtype=torch.float16,
    )

    # 2. Prepare input data
    images = prepare_images()          # [snowman_img, stop_img]
    prompts = build_prompts(processor) # [prompt_for_snowman, prompt_for_stop]

    # 3. Generate
    generations = generate_batch(
        model=model,
        processor=processor,
        images=images,
        prompts=prompts,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
    )

    # 4. Display
    for i, text in enumerate(generations):
        print(f"\n--- Sample {i} ---")
        print(text)


if __name__ == "__main__":
    main()
