from __future__ import annotations

from functools import wraps
from typing import Callable, List, Optional, Sequence

from benchmark import SampleRow


def _normalize_limit(limit_value) -> Optional[int]:
    if limit_value is None:
        return None
    try:
        limit = int(limit_value)
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else 0


def _format_phase_times(row: SampleRow) -> str:
    if not row.phase_times:
        return "phases: n/a"
    parts = [f"{name}: {dur:.3f}s" for name, dur in sorted(row.phase_times.items())]
    return " | ".join(parts)


def _print_sample(row: SampleRow, show_phases: bool):
    print(f"\n=== Sample {row.idx} ===")
    print(f"Q:  {row.user_prompt}")
    print(f"→ Model: {row.pred}")
    if row.gt != "":
        print(f"→ GT:    {row.gt}")
    parts: List[str] = []
    if show_phases:
        phase_str = _format_phase_times(row)
        parts.append(phase_str)
    parts.append(f"total: {row.t_total_s:.3f}s")
    parts.append(f"toks/s: {row.tokens_per_s:.1f}")
    print(" | ".join(parts))

def make_sample_logger(
    *,
    per_sample: bool,
    examples_n,
    show_phases: bool,
) -> Callable[[Callable[..., Sequence[SampleRow]]], Callable[..., List[SampleRow]]]:
    limit = _normalize_limit(examples_n)

    def decorator(fn: Callable[..., Sequence[SampleRow]]):
        @wraps(fn)
        def wrapper(*args, **kwargs) -> List[SampleRow]:
            rows = list(fn(*args, **kwargs))  # ensures we can iterate multiple times
            for row in rows:
                if per_sample:
                    if limit == 0:
                        continue
                    if limit is not None and row.idx >= limit:
                        continue
                    _print_sample(row, show_phases)
            return rows

        return wrapper

    return decorator
