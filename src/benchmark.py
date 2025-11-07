from __future__ import annotations
import csv, json, os, platform, subprocess, time, math
from dataclasses import dataclass, asdict, field
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar
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

T = TypeVar("T")

@dataclass
class SampleRow:
    idx: int
    user_prompt: str
    gt: str
    pred: str
    n_images: int
    input_tokens: int
    output_tokens: int
    t_total_s: float
    tokens_per_s: float  # output_tokens / total wall duration
    phase_times: Dict[str, float] = field(default_factory=dict)

class PhaseTimer:
    """Per-sample timer that can be used via start/stop, context manager, or decorator."""
    def __init__(self):
        self._t0: Dict[str, int] = {}
        self.elapsed_s: Dict[str, float] = {}

    def start(self, name: str):
        self._t0[name] = _now_ns()

    def stop(self, name: str):
        t0 = self._t0.pop(name, None)
        if t0 is None:
            return
        self.elapsed_s[name] = self.elapsed_s.get(name, 0.0) + _ns_to_s(_now_ns() - t0)

    def phase(self, name: str):
        """Context manager helper: with timer.phase(\"encode\"): ..."""
        class _PhaseCtx:
            def __init__(self, timer: PhaseTimer, phase_name: str):
                self._timer = timer
                self._name = phase_name

            def __enter__(self):
                self._timer.start(self._name)

            def __exit__(self, exc_type, exc, tb):
                self._timer.stop(self._name)
                return False
        return _PhaseCtx(self, name)

    def measure(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Decorator factory to time arbitrary callables."""
        def decorator(fn: Callable[..., T]) -> Callable[..., T]:
            @wraps(fn)
            def wrapped(*args, **kwargs) -> T:
                with self.phase(name):
                    return fn(*args, **kwargs)
            return wrapped
        return decorator

    def total(self, phases: Iterable[str] | None = None) -> float:
        names = tuple(phases) if phases is not None else tuple(self.elapsed_s.keys())
        return sum(self.elapsed_s.get(name, 0.0) for name in names)

def build_generation_stats(
    timer: PhaseTimer,
    n_images: int,
    input_tokens: int,
    output_tokens: int,
    phases: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """
    Normalize raw timer readings into the stats dict expected by benchmarking flows.
    """
    total_s = timer.total(phases)
    tokens_per_s = (output_tokens / total_s) if total_s > 0 else float("nan")
    return {
        "phase_times": dict(timer.elapsed_s),
        "n_images": n_images,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "t_total_s": total_s,
        "tokens_per_s": tokens_per_s,
    }

def aggregate(samples: List[SampleRow]) -> Dict[str, Any]:
    """Compute summary stats focusing on total latency and throughput."""
    if not samples:
        return {}

    def stats(xs: List[float]) -> Dict[str, float]:
        if not xs:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": sum(xs) / len(xs),
            "p50": _pctl(xs, 0.50),
            "p95": _pctl(xs, 0.95),
            "min": min(xs),
            "max": max(xs),
            "n": len(xs),
        }

    total_times = [s.t_total_s for s in samples]
    output_tokens = [float(s.output_tokens) for s in samples]
    tokens_per_s = [s.tokens_per_s for s in samples if not math.isnan(s.tokens_per_s)]

    return {
        "count": len(samples),
        "latency_s": stats(total_times),
        "throughput": {
            "tokens_per_s": stats(tokens_per_s),
            "avg_output_tokens": sum(output_tokens) / len(output_tokens),
            "samples_per_s": len(samples) / sum(total_times) if sum(total_times) > 0 else 0.0,
        },
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
            latency = summary.get("latency_s")
            if latency:
                with open(self.paths["phases_csv"], "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["metric", "mean_s", "p50_s", "p95_s", "min_s", "max_s", "n"])
                    w.writerow(["total", latency["mean"], latency["p50"], latency["p95"],
                                latency["min"], latency["max"], latency["n"]])

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
    latency = agg.get("latency_s", {})
    if latency:
        print(f"total latency mean(s): {latency['mean']:.3f} | p50: {latency['p50']:.3f} | p95: {latency['p95']:.3f}")
    th = agg.get("throughput", {})
    if th:
        print(f"avg output tokens: {th.get('avg_output_tokens', 0.0):.2f}")
        print(f"samples/sec (wall): {th.get('samples_per_s', 0.0):.3f}")
        tp = th.get("tokens_per_s", {})
        if tp:
            print(f"output tokens/sec (wall): mean {tp['mean']:.1f} | p50 {tp['p50']:.1f} | p95 {tp['p95']:.1f}")
    if show_phase_table:
        print("\n[info] Per-phase breakdown is omitted in the lean benchmark mode.")
