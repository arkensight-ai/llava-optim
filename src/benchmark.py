from __future__ import annotations
import csv, json, os, platform, subprocess, time
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import torch

def _now_ns() -> int:
    return time.perf_counter_ns()

def _ns_to_s(ns: int) -> float:
    return ns / 1e9

def _pctl(xs: List[float], q: float) -> float:
    if not xs: return 0.0
    xs = sorted(xs)
    k = (len(xs)-1) * q
    f = int(k)
    c = min(f+1, len(xs)-1)
    if f == c: return xs[f]
    return xs[f] + (k - f) * (xs[c] - xs[f])

@dataclass
class SampleRow:
    idx: int
    user_prompt: str
    gt: str
    pred: str
    n_images: int
    input_tokens: int
    output_tokens: int
    t_encode_s: float
    t_generate_s: float
    t_decode_s: float
    t_total_s: float
    tokens_per_s: float  # output_tokens / t_generate_s (wall)

class PhaseTimer:
    """Simple per-sample phase timer container."""
    def __init__(self):
        self._t0: Dict[str, int] = {}
        self.elapsed_s: Dict[str, float] = {}

    def start(self, name: str):
        self._t0[name] = _now_ns()

    def stop(self, name: str):
        t0 = self._t0.pop(name, None)
        if t0 is None: return
        self.elapsed_s[name] = self.elapsed_s.get(name, 0.0) + _ns_to_s(_now_ns() - t0)

def aggregate(samples: List[SampleRow]) -> Dict[str, Any]:
    """Compute summary stats across samples."""
    if not samples:
        return {}
    def col(fn):
        return [fn(s) for s in samples]
    def stats(xs: List[float]) -> Dict[str, float]:
        return {
            "mean": sum(xs)/len(xs) if xs else 0.0,
            "p50": _pctl(xs, 0.50),
            "p95": _pctl(xs, 0.95),
            "min": min(xs) if xs else 0.0,
            "max": max(xs) if xs else 0.0,
            "n": len(xs),
        }

    t_encode = col(lambda s: s.t_encode_s)
    t_generate = col(lambda s: s.t_generate_s)
    t_decode = col(lambda s: s.t_decode_s)
    t_total  = col(lambda s: s.t_total_s)
    out_tokens = col(lambda s: float(s.output_tokens))
    toks_per_s = col(lambda s: s.tokens_per_s if s.tokens_per_s == s.tokens_per_s else 0.0) # NaN-safe

    return {
        "count": len(samples),
        "phases": {
            "encode_s": stats(t_encode),
            "generate_s": stats(t_generate),
            "decode_s": stats(t_decode),
            "total_s": stats(t_total),
        },
        "throughput": {
            "output_tokens_per_s": stats(toks_per_s),
            "avg_output_tokens": sum(out_tokens)/len(out_tokens) if samples else 0.0,
            "samples_per_s": len(samples) / sum(t_total) if sum(t_total) > 0 else 0.0,
        }
    }

def collect_env() -> Dict[str, Any]:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [],
        "git_commit": None,
    }
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        pass
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            info["gpus"].append({
                "index": i,
                "name": prop.name,
                "total_mem_gb": round(prop.total_memory / (1024**3), 2),
                "cc": f"{prop.major}.{prop.minor}",
            })
    return info

class BenchmarkWriter:
    def __init__(self, out_dir: str, save_cfg: Dict[str, bool]):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.paths = {
            "samples_jsonl": os.path.join(out_dir, "samples.jsonl"),
            "summary_json":  os.path.join(out_dir, "summary.json"),
            "phases_csv":    os.path.join(out_dir, "phases.csv"),
            "hardware_json": os.path.join(out_dir, "hardware.json"),
        }
        self.save_cfg = save_cfg

    def append_sample(self, row: SampleRow):
        if not self.save_cfg.get("samples_jsonl", True): return
        with open(self.paths["samples_jsonl"], "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    def write_summary(self, summary: Dict[str, Any], meta: Dict[str, Any]):
        if self.save_cfg.get("summary_json", True):
            payload = {"summary": summary, "meta": meta}
            with open(self.paths["summary_json"], "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        if self.save_cfg.get("phases_csv", True):
            # Flatten phases for a quick CSV
            phases = summary.get("phases", {})
            with open(self.paths["phases_csv"], "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["phase", "mean_s", "p50_s", "p95_s", "min_s", "max_s", "n"])
                for phase, st in phases.items():
                    w.writerow([phase, st["mean"], st["p50"], st["p95"], st["min"], st["max"], st["n"]])

    def write_hardware(self, env: Dict[str, Any]):
        if not self.save_cfg.get("hardware_json", True): return
        with open(self.paths["hardware_json"], "w", encoding="utf-8") as f:
            json.dump(env, f, indent=2)

def print_aggregates(agg: Dict[str, Any], show_phase_table: bool):
    if not agg: 
        print("No samples collected.")
        return
    print("\n=== Aggregates ===")
    print(f"samples: {agg['count']}")
    th = agg["throughput"]
    print(f"avg output tokens: {th['avg_output_tokens']:.2f}")
    print(f"samples/sec (wall): {th['samples_per_s']:.3f}")
    print(f"output tokens/sec (wall, avg of per-sample): {th['output_tokens_per_s']['mean']:.1f}")
    if show_phase_table:
        print("\nphase       |   mean(s) |   p50(s) |   p95(s) |   min |   max |   n")
        print("------------+----------:|---------:|---------:|------:|------:|----:")
        for name, st in agg["phases"].items():
            print(f"{name:<11} | {st['mean']:9.3f} | {st['p50']:8.3f} | {st['p95']:8.3f} |"
                  f" {st['min']:5.3f} | {st['max']:5.3f} | {st['n']:4d}")
