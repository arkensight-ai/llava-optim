# src/debug_utils.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import os, re
import numpy as np

# -------- Config & controller --------

@dataclass
class DebugDetail:
    seq_report: bool = True
    io_schema: bool = False
    onnx_dtypes: bool = True
    kv_shapes: bool = False
    gen_summary: bool = True

@dataclass
class OnnxDebugCfg:
    chunk_size: int = 512
    providers: Tuple[str, ...] = ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert_fp16_ratio: float = 0.80

@dataclass
class DebugCfg:
    enabled: bool = False
    # IMPORTANT: use default_factory for mutable defaults
    detail: DebugDetail = field(default_factory=DebugDetail)
    onnx: OnnxDebugCfg = field(default_factory=OnnxDebugCfg)

def _get(node, key, default):
    try:
        if node is None: return default
        v = getattr(node, key)
        return v if v is not None else default
    except Exception:
        return default

class DebugController:
    """
    Hydra-driven debug switch with env fallback.
    Pass the whole Hydra cfg (has .verbosity) or None to use env-only.
    """
    def __init__(self, cfg: Optional[Any] = None):
        if cfg is not None and hasattr(cfg, "verbosity"):
            v = cfg.verbosity
            enabled = bool(getattr(v, "debug", False))
            dd = getattr(v, "debug_detail", None)
            onx = getattr(v, "onnx", None)
        else:
            enabled, dd, onx = False, None, None

        # env fallback (LLO_DEBUG=1 forces on)
        if os.getenv("LLO_DEBUG", "0") not in ("0", "false", "False"):
            enabled = True

        self.enabled = enabled
        self.detail = DebugDetail(
            seq_report = _get(dd, "seq_report", True),
            io_schema  = _get(dd, "io_schema", False),
            onnx_dtypes= _get(dd, "onnx_dtypes", True),
            kv_shapes  = _get(dd, "kv_shapes", False),
            gen_summary= _get(dd, "gen_summary", True),
        )
        self.onnx = OnnxDebugCfg(
            chunk_size = int(_get(onx, "chunk_size", 512)),
            providers  = tuple(_get(onx, "providers", ("CUDAExecutionProvider","CPUExecutionProvider"))),
            assert_fp16_ratio = float(_get(onx, "assert_fp16_ratio", 0.80)),
        )

    def log(self, msg: str):
        if self.enabled:
            print(msg)

# -------- Back-compat env helper (used by your current inference.py) --------
def env_debug_enabled() -> bool:
    return os.getenv("LLO_DEBUG", "0") not in ("0", "false", "False")

# -------- Shared helpers (framework-agnostic) --------

def count_tiles_from_pixel_values(pv) -> int:
    if pv is None: return 0
    shp = tuple(getattr(pv, "shape", ()))
    if len(shp) == 5:     # (B, T, C, H, W)
        return int(shp[1])
    if len(shp) == 4:     # (B*T, C, H, W) with B==1
        return int(shp[0])
    return 0

def pretty_kv_shapes(past: Any) -> str:
    try:
        first = past[0]
    except Exception:
        return "[]"
    try:
        k, v = first[:2]
        kd = getattr(k, "dtype", None)
        vd = getattr(v, "dtype", None)
        ks = getattr(k, "shape", None)
        vs = getattr(v, "shape", None)
        return f"K:{ks}/{kd} V:{vs}/{vd}"
    except Exception:
        return type(past).__name__

def debug_seq_report(
    dbg: DebugController,
    *,
    provider: str,
    text_len: Optional[int],
    vision_tokens: int,
    tiles: int,
    hidden_size: Optional[int],
    heads: Optional[int],
    layers: Optional[int],
):
    if not (dbg.enabled and dbg.detail.seq_report):
        return
    L = (text_len or 0) - 1 + (vision_tokens or 0) if text_len else (vision_tokens or 0)
    print(f"[debug/{provider}] text={text_len} vision={vision_tokens} tiles={tiles} "
          f"merged_len={L} hidden={hidden_size} heads={heads} layers={layers}")
    if L and heads and layers:
        # rough prefill FP16 attention intermediates size
        per_layer_mib = (heads * (L**2) * 2) / (1024**2)
        total_gib = (per_layer_mib * layers) / 1024
        print(f"[debug/{provider}] rough prefill activations ≈ {per_layer_mib:,.0f} MiB/layer, "
              f"total ≈ {total_gib:,.2f} GiB")

# -------- ONNX helpers (import on demand) --------

def summarize_onnx_dtypes(onnx_path: str) -> Dict[str,int]:
    import onnx
    counts: Dict[str,int] = {}
    model = onnx.load(onnx_path)
    for init in model.graph.initializer:
        dt = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[init.data_type].__name__
        counts[dt] = counts.get(dt, 0) + 1
    return counts

def print_onnx_io(session, tag: str, dbg: DebugController):
    if not (dbg.enabled and dbg.detail.io_schema):
        return
    prov = getattr(session, "get_providers", lambda: [])()
    print(f"[debug/ort] {tag}: providers={prov}")
    for io in session.get_inputs():
        print(f"  in  {io.name:28s} type={io.type} shape={io.shape}")
    for io in session.get_outputs():
        print(f"  out {io.name:28s} type={io.type} shape={io.shape}")

def onnx_in_out_names(session) -> Tuple[Dict[str,str], Dict[str,str]]:
    def _find_inp(key: str) -> Optional[str]:
        k = key.lower()
        for x in session.get_inputs():
            if k in x.name.lower():
                return x.name
        return None
    def find(keys: List[str]) -> Dict[str,str]:
        out = {}
        for k in keys:
            nm = _find_inp(k)
            if nm: out[k] = nm
        return out
    return find(["inputs_embeds","attention_mask","position_ids","cache_position"]), {}

def pkv_io_names(session) -> Tuple[List[str], List[str]]:
    ins  = [i.name for i in session.get_inputs()]
    outs = [o.name for o in session.get_outputs()]
    in_p  = [n for n in ins  if re.search(r"past.*(key|value)|present|k_.*|v_.*|cache", n, re.I)]
    out_p = [n for n in outs if re.search(r"present.*(key|value)|out_past|k_.*|v_.*|cache", n, re.I)]
    return in_p, out_p

def prefill_in_chunks_decoder(
    session,
    *,
    inputs_embeds_merged: np.ndarray,   # (1, L, H) fp16 preferred
    attention_mask_full: np.ndarray,    # (1, L)
    position_ids_full: np.ndarray,      # (1, L)
    chunk_size: int,
    dbg: DebugController,
    pkv_init: Optional[Dict[str, np.ndarray] | List[np.ndarray]] = None,
):
    """
    Incrementally prefill decoder; returns (last_logits, pkv_list, L).
    Some exports REQUIRE past_key_values.* even for the first chunk; pass them via pkv_init.
    """
    in_names, _ = onnx_in_out_names(session)
    in_pkv, out_pkv = pkv_io_names(session)
    assert "inputs_embeds" in in_names and "attention_mask" in in_names and "position_ids" in in_names, \
        "Decoder ONNX must accept inputs_embeds, attention_mask, position_ids"

    # Prepare PKV in the exact input order if provided as dict
    pkv = None
    if pkv_init is not None:
        if isinstance(pkv_init, dict):
            pkv = [pkv_init[n] for n in in_pkv] if in_pkv else None
        else:
            pkv = pkv_init  # already a list in correct order

    L = inputs_embeds_merged.shape[1]
    last_logits = None

    for start in range(0, L, max(1, chunk_size)):
        end = min(L, start + chunk_size)
        feed = {
            in_names["inputs_embeds"]: inputs_embeds_merged[:, start:end, :],
            in_names["attention_mask"]: attention_mask_full[:, :end],
            in_names["position_ids"]:   position_ids_full[:, start:end],
        }
        if "cache_position" in in_names:
            feed[in_names["cache_position"]] = np.arange(start, end, dtype=position_ids_full.dtype)[None, :]

        # IMPORTANT: if the model REQUIRES PKV, feed either the init zeros (first chunk) or the rolling PKV.
        if in_pkv:
            if pkv is None:
                if isinstance(pkv_init, dict):
                    for name in in_pkv:
                        feed[name] = pkv_init[name]  # zero-length cache from export
                elif isinstance(pkv_init, list):
                    for name, arr in zip(in_pkv, pkv_init):
                        feed[name] = arr
                else:
                    raise ValueError("This ONNX decoder requires past_key_values.* on first call; provide pkv_init.")
            else:
                for name, arr in zip(in_pkv, pkv):
                    feed[name] = arr

        outs = session.run(None, feed)

        logits_idx = next((i for i,o in enumerate(session.get_outputs()) if "logits" in o.name.lower()), 0)
        last_logits = outs[logits_idx]

        if out_pkv:
            name_to_out = {o.name: v for o, v in zip(session.get_outputs(), outs)}
            pkv = [name_to_out[n] for n in out_pkv]
        elif in_pkv:
            pkv = outs[-len(in_pkv):]

        if dbg.enabled and dbg.detail.kv_shapes:
            print(f"[chunk prefill] {start:>5d}:{end:<5d} KV={pretty_kv_shapes(pkv)}")

    return last_logits, pkv, L

