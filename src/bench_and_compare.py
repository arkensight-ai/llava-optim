import subprocess
import os
import json
import argparse
from pathlib import Path
from tabulate import tabulate # uv add tabulate

def run_cmd(cmd, env=None):
    print(f"\n🚀 Running: {' '.join(cmd)}")
    current_env = os.environ.copy()
    if env: current_env.update(env)
    subprocess.run(cmd, env=current_env, check=True)

def collect_results(root):
    data = []
    for path in Path(root).rglob("summary.json"):
        with open(path) as f:
            c = json.load(f)
            data.append({
                "Backend": c['meta']['backend'],
                "Quant": c['meta']['quant']['name'],
                "BS": c['meta']['gen']['batch_size'],
                "Tok/s": round(c['summary']['throughput']['tokens_per_s']['mean'], 2),
                "Latency (p50)": round(c['summary']['latency_s']['p50'], 2),
            })
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./results/smolvlm_256m")
    args = parser.parse_args()
    
    os.makedirs(args.dir, exist_ok=True)
    
    # 256M Benchmark Matrix
    # name, backend, quant, batch_size, env_overrides
    tasks = [
        ("HF-FP16", "hf", "fp16", 1, {}),
        ("HF-4bit", "hf", "fp16", 8, {}),
        ("vLLM-BS1", "vllm", "fp16", 1, {"VLLM_USE_V1": "0"}),
        ("vLLM-BS8", "vllm", "fp16", 8, {"VLLM_USE_V1": "0"}),
    ]
    
    for name, backend, quant, bs, env in tasks:
        run_cmd([
            "uv", "run", "python", "./src/bench_hydra.py",
            "model=smolvlm", # Uses your smolvlm.yaml
            "csv=./data/bench_single.csv",
            f"model.backend={backend}",
            f"quant={quant}",
            f"gen.batch_size={bs}",
            f"verbosity.out_dir={os.path.join(args.dir, name)}",
            "model.gpu_memory_utilization=0.6" # 256M is tiny, 0.6 is plenty
        ], env=env)

    print("\n" + "="*40 + "\n📈 FINAL COMPARISON\n" + "="*40)
    res = collect_results(args.dir)
    print(tabulate(sorted(res, key=lambda x: x['Tok/s'], reverse=True), headers="keys"))

if __name__ == "__main__":
    main()