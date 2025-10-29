
from __future__ import annotations
import argparse, os, sys
# Ensure repo-root/src is on sys.path so `import llava` works when running this file directly.
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(FILE_DIR)  # points to .../src
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from inference import load_model, generate_single_with_stats, generate_batch_with_stats
from data_loading import prepare_inputs_from_csv
from benchmark import pretty_print_stats, write_stats_jsonl

def parse_args():
    p = argparse.ArgumentParser(description="Run LLaVA OneVision on a CSV of prompts/images, with profiling.")
    p.add_argument("--csv", type=str, required=True, help="CSV with columns: image_paths,prompt,answer. Multi-images separated by ';'.")
    p.add_argument("--model-id", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", help="HF repo id.")
    p.add_argument("--four-bit", action="store_true", help="Load model in 4-bit quantization.")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size for inference (classic parallelization).")
    p.add_argument("--max-new-tokens", type=int, default=50, help="Max new tokens to generate.")
    p.add_argument("--top-p", type=float, default=0.9, help="Top-p (used only if sampling)." )
    p.add_argument("--no-sample", action="store_true", help="Disable sampling (greedy)." )
    p.add_argument("--max-image-side", type=int, default=1280, help="Resize each image so its longest side <= this.")
    p.add_argument("--out-jsonl", type=str, default=None, help="Optional: path to write JSONL with per-sample stats.")
    return p.parse_args()

def main():
    args = parse_args()
    model, processor = load_model(model_id=args.model_id, four_bit=args.four_bit)

    images_batch, model_prompts, user_prompts, answers = prepare_inputs_from_csv(
        csv_path=args.csv,
        processor=processor,
        max_image_side=args.max_image_side,
    )

    bs = int(args.batch_size)
    for start in range(0, len(model_prompts), bs):
        end = min(start + bs, len(model_prompts))
        imgs_chunk = images_batch[start:end]
        prom_chunk = model_prompts[start:end]
        user_chunk = user_prompts[start:end]
        gt_chunk   = answers[start:end]

        # batched
        preds, stats_list = generate_batch_with_stats(
            model=model,
            processor=processor,
            images_batch=imgs_chunk,
            model_prompts=prom_chunk,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.no_sample,
            top_p=args.top_p,
            temperature=args.temperature,
        )

        for j, (pred, stats) in enumerate(zip(preds, stats_list)):
            idx = start + j
            print(f"\n=== Sample {idx} ===")
            print(f"Q:  {user_chunk[j]}")
            print(f"→ Model: {pred}")
            if gt_chunk[j] != "":
                print(f"→ GT:    {gt_chunk[j]}")
            pretty_print_stats(stats)
            if args.out_jsonl is not None:
                write_stats_jsonl(args.out_jsonl, idx, user_chunk[j], gt_chunk[j], pred, stats)

if __name__ == "__main__":
    main()
