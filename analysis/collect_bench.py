import json, glob, os, pandas as pd

rows = []
for path in glob.glob("outputs/**/summary.json", recursive=True):
    with open(path) as f:
        d = json.load(f)
    meta = d["meta"]
    agg  = d["summary"]
    rows.append({
        "run_dir": os.path.dirname(path),
        "model_id": meta["model_id"],
        "quant": meta["quant"].get("name", "fp"),
        "gen.max_new_tokens": meta["gen"]["max_new_tokens"],
        "n_samples": meta["dataset"]["n"],
        "encode_mean_s": agg["phases"]["encode_s"]["mean"],
        "generate_mean_s": agg["phases"]["generate_s"]["mean"],
        "decode_mean_s": agg["phases"]["decode_s"]["mean"],
        "total_mean_s": agg["phases"]["total_s"]["mean"],
        "tok_s_mean": agg["throughput"]["output_tokens_per_s"]["mean"],
        "samples_per_s": agg["throughput"]["samples_per_s"],
    })

df = pd.DataFrame(rows).sort_values(["model_id", "quant"])
print(df.to_string(index=False))
df.to_csv("outputs/quant_compare.csv", index=False)
