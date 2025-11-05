#!/usr/bin/env python3
"""
Download the ONNX export of LLaVA OneVision (Qwen2-0.5B-OV) at a fixed commit
and place all files from the repo's `onnx/` folder directly into a target dir.
Compatible with older huggingface_hub (no use_symlinks kw).

Usage:
  uv run python download_llava_onnx_v2.py --target models/llava-onevision-qwen2-0.5b-ov-hf
"""
import argparse
import shutil
import sys
from pathlib import Path

def main():
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("ERROR: huggingface_hub is not installed. Install with:", file=sys.stderr)
        print("  uv pip install -U huggingface_hub", file=sys.stderr)
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
                        help="HF repo id")
    parser.add_argument("--rev", default="8fb52d2643122bd84379de3b07ef185966f72cda",
                        help="Commit hash / tag / branch for reproducible fetch")
    parser.add_argument("--target", required=True,
                        help="Target directory where ONNX files will be placed (flattened)")
    parser.add_argument("--include-all", action="store_true",
                        help="If set, copy all files from onnx/ (not just *.onnx)")
    args = parser.parse_args()

    print(f"Downloading {args.repo}@{args.rev} (onnx/*) ...")
    # Keep it simple for compatibility: don't pass 'use_symlinks' (older hubs don't accept it)
    repo_dir = snapshot_download(
        repo_id=args.repo,
        revision=args.rev,
        allow_patterns=["onnx/*"],
        local_files_only=False,
    )
    repo_dir = Path(repo_dir)
    src_dir = repo_dir / "onnx"
    if not src_dir.exists():
        print(f"ERROR: Could not find 'onnx' dir inside snapshot at {repo_dir}", file=sys.stderr)
        sys.exit(2)

    target_dir = Path(args.target)
    target_dir.mkdir(parents=True, exist_ok=True)

    patterns = ["**/*"] if args.include_all else ["**/*.onnx"]
    files = []
    for pat in patterns:
        files.extend(p for p in src_dir.glob(pat) if p.is_file())

    if not files:
        print("WARNING: No files matched. (Try --include-all if you expected non-ONNX files.)")

    # copy2 follows symlinks by default; this dereferences thin copies from the cache.
    for src in files:
        dst = target_dir / src.name  # flatten filename into target dir
        shutil.copy2(src, dst)
        print(f"Copied {src.relative_to(src_dir)} -> {dst}")

    print(f"\nDone. Placed {len(files)} file(s) in: {target_dir.resolve()}")
    print("Set Hydra: model.onnx_dir=<that path>")

if __name__ == "__main__":
    main()
