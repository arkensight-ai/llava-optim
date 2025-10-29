
from __future__ import annotations
from typing import Dict, Any, Iterable, Optional
import csv, os, json

def pretty_print_stats(stats: Dict[str, Any]) -> None:
    print("  --- Profiling ---")
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

def write_stats_jsonl(path: str, sample_idx: int, question: str, gt: str, pred: str, stats: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        rec = {
            "sample": sample_idx,
            "question": question,
            "ground_truth": gt,
            "prediction": pred,
            "stats": stats,
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
