from __future__ import annotations
import os
from typing import Callable, Optional, Tuple
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
import hydra
from hydra.core.hydra_config import HydraConfig
import torch

import inference as hf_backend
import onnx_inference as onnx_backend
from data_loading import prepare_inputs_from_csv
from benchmark import SampleRow, aggregate, BenchmarkWriter, collect_env, print_aggregates
from logging_utils import make_sample_logger
from debug_utils import DebugController


def _resolve_backend(name: Optional[str]) -> Tuple[Callable[..., object], Callable[..., object]]:
    normalized = (name or "hf").lower()
    if normalized == "onnx":
        return onnx_backend.load_model, onnx_backend.generate_with_stats
    return hf_backend.load_model, hf_backend.generate_with_stats


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
    load_model_fn, generate_with_stats_fn = _resolve_backend(getattr(cfg.model, "backend", None))
    is_onnx_backend = load_model_fn is onnx_backend.load_model
    load_kwargs = {
        "model_id": cfg.model.model_id,
        "quant": cfg.quant,
    }
    if load_model_fn is onnx_backend.load_model:
        onnx_dir = getattr(cfg.model, "onnx_dir", None)
        if not onnx_dir:
            raise ValueError("cfg.model.onnx_dir must be set when using the ONNX backend.")
        load_kwargs["onnx_dir"] = onnx_dir

    model, processor = load_model_fn(**load_kwargs)

    max_image_side = int(cfg.preprocess.max_image_side)
    # if is_onnx_backend:
    #     onnx_side = int(getattr(cfg.model, "max_image_side", 384))
    #     max_image_side = min(max_image_side, onnx_side)
    #     if max_image_side < cfg.preprocess.max_image_side:
    #         print(f"[info] Clamping max_image_side to {max_image_side} for ONNX backend to reduce VRAM usage.")

    # Data (count as a run-level phase externally if you want)
    images_batch, model_prompts, user_prompts, answers = prepare_inputs_from_csv(
        processor=processor,
        csv_path=csv_path,
        max_image_side=max_image_side,
    )

    sample_logger = make_sample_logger(
        per_sample=bool(cfg.verbosity.print.per_sample),
        examples_n=cfg.verbosity.print.examples_n,
        show_phases=bool(cfg.verbosity.print.phase_table),
    )

    # Loop
    samples: list[SampleRow] = []

    bs = int(cfg.gen.batch_size)

    @sample_logger
    def _rows_from_batch(start_idx: int, preds: list[str], stats_list) -> list[SampleRow]:
        rows: list[SampleRow] = []
        for j, (pred, s) in enumerate(zip(preds, stats_list)):
            idx = start_idx + j
            phase_times = {k: float(v) for k, v in (s.get("phase_times") or {}).items()}
            row = SampleRow(
                idx=idx,
                user_prompt=user_prompts[idx],
                gt=answers[idx],
                pred=pred,
                n_images=int(s["n_images"]),
                input_tokens=int(s["input_tokens"]),
                output_tokens=int(s["output_tokens"]),
                t_total_s=float(s["t_total_s"]),
                tokens_per_s=float(s["tokens_per_s"]) if s["tokens_per_s"] == s["tokens_per_s"] else 0.0,
                phase_times=phase_times,
            )
            rows.append(row)
        return rows

    for start in range(0, len(model_prompts), bs):
        end = min(start + bs, len(model_prompts))
        imgs_chunk = images_batch[start:end]
        prom_chunk = model_prompts[start:end]

        preds, stats_list = generate_with_stats_fn(
            model=model,
            processor=processor,
            images=imgs_chunk,
            model_prompts=prom_chunk,
            max_new_tokens=int(cfg.gen.max_new_tokens),
            do_sample=bool(cfg.gen.do_sample),
            top_p=float(cfg.gen.top_p),
            temperature=float(getattr(cfg.gen, "temperature", 1.0)),
            dbg=DebugController(cfg),
        )

        rows = _rows_from_batch(start, preds, stats_list)
        samples.extend(rows)

        # Persist per-sample
        for row in rows:
            writer.append_sample(row)
            if out_jsonl_legacy:
                os.makedirs(os.path.dirname(to_absolute_path(out_jsonl_legacy)), exist_ok=True)
                with open(to_absolute_path(out_jsonl_legacy), "a", encoding="utf-8") as f:
                    f.write(OmegaConf.to_yaml({"idx": row.idx, "sample": row.__dict__}))

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
