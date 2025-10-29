from __future__ import annotations
import os
from typing import Optional
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import hydra
from hydra.core.hydra_config import HydraConfig
import torch

from inference import load_model, generate_single_with_stats
from data_loading import prepare_inputs_from_csv
from benchmark import SampleRow, aggregate, BenchmarkWriter, collect_env, print_aggregates


@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("\n=== Resolved config ===")
    print(OmegaConf.to_yaml(cfg))

    if cfg.gen.seed is not None:
        torch.manual_seed(int(cfg.gen.seed))

    # Resolve I/O
    csv_path = to_absolute_path(cfg.csv)
    out_jsonl_legacy: Optional[str] = cfg.get("out_jsonl")
    runtime_out_dir = HydraConfig.get().runtime.output_dir
    out_dir = cfg.verbosity.out_dir or runtime_out_dir
    out_dir = to_absolute_path(out_dir)


    writer = BenchmarkWriter(
        out_dir=out_dir,
        save_cfg=dict(cfg.verbosity.save),
    )

    # Model
    model, processor = load_model(
        model_id=cfg.model.model_id,
        four_bit=None,
        quant=cfg.quant,
    )

    # Data (count as a run-level phase externally if you want)
    images_batch, model_prompts, user_prompts, answers = prepare_inputs_from_csv(
        processor=processor,
        csv_path=csv_path,
        max_image_side=cfg.preprocess.max_image_side,
    )

    # Loop
    samples: list[SampleRow] = []

    for idx in range(len(model_prompts)):
        pred, s = generate_single_with_stats(
            model=model,
            processor=processor,
            images_for_prompt=images_batch[idx],
            model_prompt=model_prompts[idx],
            max_new_tokens=cfg.gen.max_new_tokens,
            do_sample=cfg.gen.do_sample,
            top_p=cfg.gen.top_p,
        )

        row = SampleRow(
            idx=idx,
            user_prompt=user_prompts[idx],
            gt=answers[idx],
            pred=pred,
            n_images=s["n_images"],
            input_tokens=int(s["input_tokens"]),
            output_tokens=int(s["output_tokens"]),
            t_encode_s=float(s["time_s"].get("encode", 0.0)),
            t_generate_s=float(s["time_s"].get("generate", 0.0)),
            t_decode_s=float(s["time_s"].get("decode", 0.0)),
            t_total_s=float(s["t_total_s"]),
            tokens_per_s=float(s["tokens_per_s"]) if s["tokens_per_s"] == s["tokens_per_s"] else 0.0,
        )
        samples.append(row)

        # Printing (verbosity)
        if cfg.verbosity.print.per_sample:
            print(f"\n=== Sample {idx} ===")
            print(f"Q:  {row.user_prompt}")
            print(f"→ Model: {row.pred}")
            if row.gt != "":
                print(f"→ GT:    {row.gt}")
            print(f"encode: {row.t_encode_s:.3f}s | generate: {row.t_generate_s:.3f}s | "
                  f"decode: {row.t_decode_s:.3f}s | total: {row.t_total_s:.3f}s | "
                  f"toks/s: {row.tokens_per_s:.1f}")

        # Show examples (first N only)
        if cfg.verbosity.print.examples_n and idx < int(cfg.verbosity.print.examples_n) and not cfg.verbosity.print.per_sample:
            print(f"\n=== Example {idx} ===")
            print(f"Q:  {row.user_prompt}")
            print(f"→ Model: {row.pred}")
            if row.gt != "":
                print(f"→ GT:    {row.gt}")

        # Persist per-sample
        writer.append_sample(row)
        if out_jsonl_legacy:
            os.makedirs(os.path.dirname(to_absolute_path(out_jsonl_legacy)), exist_ok=True)
            with open(to_absolute_path(out_jsonl_legacy), "a", encoding="utf-8") as f:
                f.write(OmegaConf.to_yaml({"idx": idx, "sample": row.__dict__}))

    # Aggregates
    agg = aggregate(samples)
    if cfg.verbosity.print.aggregates:
        print_aggregates(agg, cfg.verbosity.print.phase_table)

    # Metadata for comparison
    meta = {
        "model_id": cfg.model.model_id,
        "quant": OmegaConf.to_container(cfg.quant, resolve=True),
        "gen": OmegaConf.to_container(cfg.gen, resolve=True),
        "preprocess": OmegaConf.to_container(cfg.preprocess, resolve=True),
        "dataset": {"csv": os.path.basename(csv_path), "n": len(samples)},
    }
    writer.write_summary(agg, meta)
    writer.write_hardware(collect_env())

    print(f"\nArtifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
