import json
import os
from pathlib import Path
import pandas as pd # uv add pandas tabulate

def collect_results(root_dir="."):
    data = []
    # Search both standard outputs and multirun directories
    for path in Path(root_dir).rglob("summary.json"):
        try:
            with open(path) as f:
                content = json.load(f)
                meta, summ = content['meta'], content['summary']
                data.append({
                    "Model": meta['model_id'].split('/')[-1],
                    "Backend": meta['backend'],
                    "Quant": meta['quant']['name'],
                    "BS": meta['gen']['batch_size'],
                    "Tokens/s": round(summ['throughput']['tokens_per_s']['mean'], 2),
                    "Latency p50": round(summ['latency_s']['p50'], 2),
                    "Path": path.parent.name
                })
        except: continue
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = collect_results()
    if not df.empty:
        df = df.sort_values("Tokens/s", ascending=False)
        print("\n" + df.to_markdown(index=False))
        
        # Calculate specific SmolVLM speedup
        s = df[df['Model'].str.contains('SmolVLM')]
        if len(s[s['Backend']=='hf']) and len(s[s['Backend']=='vllm']):
            speedup = s[s['Backend']=='vllm']['Tokens/s'].max() / s[s['Backend']=='hf']['Tokens/s'].max()
            print(f"\n🚀 Max vLLM speedup over HF for SmolVLM: {speedup:.2f}x")