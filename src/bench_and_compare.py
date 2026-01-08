import subprocess, os, json, argparse, pandas as pd
from pathlib import Path
from tabulate import tabulate

def run_cmd(cmd, env=None):
    current_env = os.environ.copy()
    if env: current_env.update(env)
    subprocess.run(cmd, env=current_env, check=True)

def get_report(results_dir):
    rows = []
    for path in Path(results_dir).rglob("summary.json"):
        with open(path) as f:
            d = json.load(f)
            m, s, t = d['meta'], d['summary'], d['meta']['timing']
            rows.append({
                "Backend": m['backend'], "BS": m['batch_size'],
                "Inference (s)": round(t['total_inference_s'], 1),
                "ms/Image": round(t['latency_per_img_s'] * 1000, 1),
                "System TPS": round(s['throughput']['tokens_per_s']['mean'] * m['batch_size'], 1),
                "p50 Latency (ms)": round(s['latency_s']['p50'] * 1000, 1),
            })
    return pd.DataFrame(rows).sort_values("System TPS", ascending=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--tokens", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.dir, exist_ok=True)

    # Note: Reduced BS32 to BS16 for HF to prevent OOM on T4
    tasks = [
        ("HF-BS1", "hf", 1, {}),
        ("HF-BS8", "hf", 8, {}),
        ("vLLM-BS1", "vllm", 1, {"VLLM_USE_V1": "0"}),
        ("vLLM-BS8", "vllm", 8, {"VLLM_USE_V1": "0"}),
        ("vLLM-BS32", "vllm", 32, {"VLLM_USE_V1": "0"}),
        ("vLLM-BS64", "vllm", 64, {"VLLM_USE_V1": "0"}),
    ]

    for name, backend, bs, env in tasks:
        run_cmd(["uv", "run", "python", "./src/bench_hydra.py", 
                 "model=smolvlm", f"csv={args.csv}", f"gen.max_new_tokens={args.tokens}",
                 f"model.backend={backend}", f"gen.batch_size={bs}", 
                 f"verbosity.out_dir={os.path.join(args.dir, name)}"], env=env)

    print("\n" + "="*70 + "\n📈 PRODUCTION PERFORMANCE REPORT (Prompt-to-Finish)\n" + "="*70)
    print(tabulate(get_report(args.dir), headers="keys", tablefmt="fancy_grid", showindex=False))

if __name__ == "__main__": main()