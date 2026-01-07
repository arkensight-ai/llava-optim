import argparse
import os
import sys
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Ensure repo-root/src is on sys.path
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
if FILE_DIR not in sys.path:
    sys.path.insert(0, FILE_DIR)

from benchmark import (
    PhaseTimer, build_generation_stats, aggregate, 
    BenchmarkWriter, collect_env, print_aggregates, SampleRow
)
from data_loading import prepare_inputs_from_csv
from inference import load_model_vllm

def run_hf_benchmark(args, images_batch, model_prompts, user_prompts, answers):
    print(f"\n[HF] Loading model: {args.model_id}")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()

    samples = []
    # Warmup
    print("[HF] Warming up...")
    for _ in range(2):
        _ = model.generate(**processor(text=model_prompts[0], images=images_batch[0], return_tensors="pt").to(model.device), max_new_tokens=10)
    torch.cuda.synchronize()

    print("[HF] Starting Benchmark...")
    for i in range(len(model_prompts)):
        timer = PhaseTimer()
        imgs, prompt = images_batch[i], model_prompts[i]
        
        with timer.phase("total"):
            # 1. Encode
            with timer.phase("encode"):
                inputs = processor(text=prompt, images=imgs, return_tensors="pt").to(model.device)
            
            # 2. Generate (with Sync)
            torch.cuda.synchronize()
            with timer.phase("generate"):
                output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
                torch.cuda.synchronize()
            
            # 3. Decode
            with timer.phase("decode"):
                generated_text = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()

        # Consistent token counting
        in_tokens = len(processor.tokenizer.encode(prompt))
        out_tokens = len(processor.tokenizer.encode(generated_text))
        
        stats = build_generation_stats(timer, len(imgs), in_tokens, out_tokens)
        row = SampleRow(i, user_prompts[i], answers[i], generated_text, len(imgs), in_tokens, out_tokens, stats['t_total_s'], stats['tokens_per_s'], stats['phase_times'])
        samples.append(row)
        print(f"Sample {i} | Tokens/s: {stats['tokens_per_s']:.2f}")

    return samples

def run_vllm_benchmark(args, images_batch, model_prompts, user_prompts, answers):
    print(f"\n[vLLM] Loading model: {args.model_id}")
    # Use the helper from your inference.py
    llm, processor = load_model_vllm(args.model_id, gpu_memory_utilization=args.gpu_util)
    
    from vllm import SamplingParams
    sampling_params = SamplingParams(max_tokens=args.max_new_tokens, temperature=0.0)

    samples = []
    print("[vLLM] Starting Benchmark...")
    for i in range(len(model_prompts)):
        timer = PhaseTimer()
        imgs, prompt = images_batch[i], model_prompts[i]
        
        vllm_inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": imgs if len(imgs) > 1 else imgs[0]},
        }

        with timer.phase("total"):
            # vLLM handles internal timing, but we wrap the call for wall-clock parity
            with timer.phase("generate"):
                outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)
            
            generated_text = outputs[0].outputs[0].text.strip()

        in_tokens = len(processor.tokenizer.encode(prompt))
        out_tokens = len(processor.tokenizer.encode(generated_text))
        
        stats = build_generation_stats(timer, len(imgs), in_tokens, out_tokens)
        row = SampleRow(i, user_prompts[i], answers[i], generated_text, len(imgs), in_tokens, out_tokens, stats['t_total_s'], stats['tokens_per_s'], stats['phase_times'])
        samples.append(row)
        print(f"Sample {i} | Tokens/s: {stats['tokens_per_s']:.2f}")

    return samples

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--model-id", type=str, default="HuggingFaceTB/SmolVLM-Instruct")
    p.add_argument("--backend", type=str, choices=["hf", "vllm"], required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--max-image-side", type=int, default=1280)
    p.add_argument("--gpu-util", type=float, default=0.8)
    p.add_argument("--out-dir", type=str, default="results/smolvlm_bench")
    args = p.parse_args()

    # Shared Data Loading
    # Note: Using AutoProcessor to ensure compatibility with SmolVLM
    temp_processor = AutoProcessor.from_pretrained(args.model_id)
    images_batch, model_prompts, user_prompts, answers = prepare_inputs_from_csv(
        args.csv, temp_processor, args.max_image_side
    )

    if args.backend == "hf":
        samples = run_hf_benchmark(args, images_batch, model_prompts, user_prompts, answers)
    else:
        samples = run_vllm_benchmark(args, images_batch, model_prompts, user_prompts, answers)

    # Aggregation and Reporting
    agg = aggregate(samples)
    print_aggregates(agg, show_phase_table=True)
    
    writer = BenchmarkWriter(os.path.join(args.out_dir, args.backend), {"summary_json": True, "samples_jsonl": True})
    for s in samples: writer.append_sample(s)
    writer.write_summary(agg, {"model": args.model_id, "backend": args.backend})
    writer.write_hardware(collect_env())

if __name__ == "__main__":
    main()