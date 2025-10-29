from __future__ import annotations
import os, sys
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import hydra
import torch

# Local imports (since we're executing the file directly from repo root)
from inference import load_model, generate_single_with_stats
from data_loading import prepare_inputs_from_csv
from benchmark import pretty_print_stats, write_stats_jsonl


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("\n=== Resolved config ===")
    print(OmegaConf.to_yaml(cfg))

    if cfg.gen.seed is not None:
        torch.manual_seed(int(cfg.gen.seed))

    csv_path = to_absolute_path(cfg.csv)
    out_jsonl: Optional[str] = cfg.get("out_jsonl")
    if out_jsonl:
        out_jsonl = to_absolute_path(out_jsonl)

    model, processor = load_model(
        model_id=cfg.model.model_id,
        four_bit=None,
        quant=cfg.quant,
    )

    images_batch, model_prompts, user_prompts, answers = prepare_inputs_from_csv(
        processor=processor,
        csv_path=csv_path,
        max_image_side=cfg.preprocess.max_image_side,
    )

    for idx in range(len(model_prompts)):
        pred, stats = generate_single_with_stats(
            model=model,
            processor=processor,
            images_for_prompt=images_batch[idx],
            model_prompt=model_prompts[idx],
            max_new_tokens=cfg.gen.max_new_tokens,
            do_sample=cfg.gen.do_sample,
            top_p=cfg.gen.top_p,
        )

        print(f"\n=== Sample {idx} ===")
        print(f"Q:  {user_prompts[idx]}")
        print(f"→ Model: {pred}")
        if answers[idx] != "":
            print(f"→ GT:    {answers[idx]}")
        pretty_print_stats(stats)

        if out_jsonl:
            write_stats_jsonl(out_jsonl, idx, user_prompts[idx], answers[idx], pred, stats)


if __name__ == "__main__":
    main()
