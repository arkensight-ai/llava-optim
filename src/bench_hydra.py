from __future__ import annotations
import os, time, hydra, torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from inference import load_model, load_model_vllm, generate_with_stats, generate_with_stats_vllm
from data_loading import prepare_inputs_from_csv
from benchmark import SampleRow, aggregate, BenchmarkWriter, collect_env, print_aggregates

@hydra.main(config_path="../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    full_start = time.perf_counter()
    backend = str(getattr(cfg.model, "backend", "hf")).lower()
    
    load_start = time.perf_counter()
    if backend == "hf":
        model, processor = load_model(cfg.model.model_id, cfg.quant)
        generate_fn = generate_with_stats
    else:
        model, processor = load_model_vllm(cfg.model.model_id, cfg.quant, 
                                          max_model_len=2048, gpu_memory_utilization=0.7)
        generate_fn = generate_with_stats_vllm
    load_time = time.perf_counter() - load_start

    images_batch, model_prompts, user_prompts, answers = prepare_inputs_from_csv(
        processor=processor, csv_path=to_absolute_path(cfg.csv), max_image_side=cfg.preprocess.max_image_side)

    samples: list[SampleRow] = []
    bs = int(cfg.gen.batch_size)
    
    inf_start = time.perf_counter()
    for start in range(0, len(model_prompts), bs):
        end = min(start + bs, len(model_prompts))
        p, s_l = generate_fn(model=model, processor=processor, images=images_batch[start:end], 
                             model_prompts=model_prompts[start:end], max_new_tokens=cfg.gen.max_new_tokens)
        for j, (pred, s) in enumerate(zip(p, s_l)):
            idx = start + j
            samples.append(SampleRow(idx=idx, user_prompt=user_prompts[idx], gt=answers[idx], pred=pred,
                                     n_images=s["n_images"], input_tokens=s["input_tokens"],
                                     output_tokens=s["output_tokens"], t_total_s=s["t_total_s"],
                                     tokens_per_s=s["tokens_per_s"], phase_times=s["phase_times"]))
    
    inference_only_time = time.perf_counter() - inf_start
    full_time = time.perf_counter() - full_start

    out_dir = to_absolute_path(cfg.verbosity.out_dir or HydraConfig.get().runtime.output_dir)
    writer = BenchmarkWriter(out_dir=out_dir, save_cfg=dict(cfg.verbosity.save))
    for row in samples: writer.append_sample(row)
    
    agg = aggregate(samples)
    meta = {
        "model_id": cfg.model.model_id, "backend": backend, "batch_size": bs,
        "timing": {
            "total_wall_s": full_time, 
            "load_time_s": load_time,
            "total_inference_s": inference_only_time,
            "latency_per_img_s": inference_only_time / len(samples)
        }
    }
    writer.write_summary(agg, meta)
    print_aggregates(agg, False)
    print(f"\n✅ Load: {load_time:.1f}s | Inference: {inference_only_time:.1f}s")

if __name__ == "__main__": main()