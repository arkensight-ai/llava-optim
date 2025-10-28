import os, time, argparse, yaml, math, json, uuid
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
import pandas as pd
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig

def now_iso():
    return time.strftime("%Y%m%dT%H%M%S")

def set_seed(seed: int):
    if seed is None: return
    import random, numpy as np
    random.seed(seed); torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def normalize_ans(s: str) -> str:
    if s is None: return ""
    return " ".join(s.strip().lower().split())

def token_f1(pred: str, gold: str) -> float:
    ps = normalize_ans(pred).split(); gs = normalize_ans(gold).split()
    if len(ps)==0 and len(gs)==0: return 1.0
    if len(ps)==0 or len(gs)==0: return 0.0
    common = {}; match = 0
    for w in gs: common[w]=common.get(w,0)+1
    for w in ps:
        if common.get(w,0)>0:
            match += 1; common[w]-=1
    if match==0: return 0.0
    prec, rec = match/len(ps), match/len(gs)
    return 2*prec*rec/(prec+rec)

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)

@dataclass
class BenchConfig:
    model_id: str; dtype: str; device_map: str; use_flash_attn_2: bool; attn_implementation: str
    quant_mode: str; bnb_4bit_compute_dtype: str; bnb_4bit_quant_type: str; bnb_4bit_use_double_quant: bool
    max_new_tokens: int; temperature: float; top_p: float; num_beams: int; repetition_penalty: float
    csv_path: str; image_delimiter: str; image_base: str; prompt_column: str; answer_column: str; image_paths_column: str
    batch_size: int; pad_side_left: bool; seed: Optional[int]
    out_dir: str; save_generations_csv: bool; save_jsonl: bool
    @staticmethod
    def from_yaml(path: str) -> "BenchConfig":
        with open(path, "r") as f: y = yaml.safe_load(f)
        return BenchConfig(
            model_id=y["model"]["model_id"], dtype=y["model"]["dtype"],
            device_map=y["model"].get("device_map","auto"),
            use_flash_attn_2=bool(y["model"].get("use_flash_attn_2", False)),
            attn_implementation=y["model"].get("attn_implementation","sdpa"),
            quant_mode=y["quantization"]["mode"],
            bnb_4bit_compute_dtype=y["quantization"].get("bnb_4bit_compute_dtype","bf16"),
            bnb_4bit_quant_type=y["quantization"].get("bnb_4bit_quant_type","nf4"),
            bnb_4bit_use_double_quant=bool(y["quantization"].get("bnb_4bit_use_double_quant", True)),
            max_new_tokens=int(y["generation"]["max_new_tokens"]),
            temperature=float(y["generation"]["temperature"]),
            top_p=float(y["generation"]["top_p"]), num_beams=int(y["generation"]["num_beams"]),
            repetition_penalty=float(y["generation"]["repetition_penalty"]),
            csv_path=y["data"]["csv_path"], image_delimiter=y["data"]["image_delimiter"],
            image_base=y["data"].get("image_base",""),
            prompt_column=y["data"]["prompt_column"], answer_column=y["data"]["answer_column"],
            image_paths_column=y["data"]["image_paths_column"],
            batch_size=int(y["inference"]["batch_size"]), pad_side_left=bool(y["inference"]["pad_side_left"]),
            seed=y["inference"].get("seed", None),
            out_dir=y["logging"]["out_dir"],
            save_generations_csv=bool(y["logging"].get("save_generations_csv", True)),
            save_jsonl=bool(y["logging"].get("save_jsonl", True)),
        )

def load_images(paths: List[str]) -> List[Image.Image]:
    ims = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        ims.append(im)
    return ims

def build_model_and_processor(cfg: BenchConfig):
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map.get(cfg.dtype, torch.bfloat16)

    model_kwargs = {}
    if cfg.quant_mode in ("bnb-4bit","bnb-8bit"):
        quant_config = BitsAndBytesConfig(
            load_in_4bit=(cfg.quant_mode=="bnb-4bit"),
            load_in_8bit=(cfg.quant_mode=="bnb-8bit"),
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type if cfg.quant_mode=="bnb-4bit" else None,
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bnb_4bit_compute_dtype=="bf16" else torch.float16,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant if cfg.quant_mode=="bnb-4bit" else None,
        )
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["torch_dtype"] = torch_dtype
        model_kwargs["device_map"] = cfg.device_map
    else:
        model_kwargs["dtype"] = torch_dtype
        model_kwargs["device_map"] = cfg.device_map
    if cfg.use_flash_attn_2:
        model_kwargs["use_flash_attention_2"] = True
        model_kwargs["attn_implementation"] = "flash_attention_2"
    else:
        model_kwargs["attn_implementation"] = cfg.attn_implementation

    processor = AutoProcessor.from_pretrained(cfg.model_id, use_fast=True)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(cfg.model_id, **model_kwargs)
    if cfg.pad_side_left and hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor

def row_to_messages(row, cfg: BenchConfig):
    images_raw = str(row[cfg.image_paths_column])
    image_paths = [p.strip() for p in images_raw.split(cfg.image_delimiter)] if images_raw else []
    if cfg.image_base:
        image_paths = [p if os.path.isabs(p) or p.startswith("http") else os.path.join(cfg.image_base, p) for p in image_paths]

    # Load PIL images and embed them directly in the content blocks
    imgs = []
    for p in image_paths:
        from PIL import Image as _PILImage
        imgs.append(_PILImage.open(p).convert("RGB"))

    content = [{"type": "image", "image": im} for im in imgs]
    content.append({"type": "text", "text": str(row[cfg.prompt_column])})
    messages = [{"role": "user", "content": content}]
    return messages


def batch_messages(rows, cfg: BenchConfig):
    msg_list = []
    for _, row in rows.iterrows():
        msgs = row_to_messages(row, cfg)
        msg_list.append(msgs)
    return msg_list


def generate_batch(model, processor, convs, cfg: BenchConfig):
    t0 = time.perf_counter()
    inputs = processor.apply_chat_template(
        convs,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        padding_side="left" if cfg.pad_side_left else "right",
        # IMPORTANT: no `images=` here on TF >= 4.49
    )
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    inputs = inputs.to(model.device, dtype_map.get(cfg.dtype, torch.bfloat16))
    t1 = time.perf_counter()

    if torch.cuda.is_available(): torch.cuda.synchronize()
    gen_start = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            num_beams=cfg.num_beams,
            repetition_penalty=cfg.repetition_penalty,
            do_sample=(cfg.temperature or 0.0) > 0.0 and cfg.num_beams == 1,
        )
    if torch.cuda.is_available(): torch.cuda.synchronize()
    gen_end = time.perf_counter()

    texts = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    input_len = inputs["input_ids"].size(1)
    new_tokens_each = [int(out_ids.size(1) - input_len) for _ in range(out_ids.size(0))]
    return {
        "texts": texts,
        "preproc_ms": (t1 - t0) * 1000,
        "generate_ms": (gen_end - gen_start) * 1000,
        "new_tokens_each": new_tokens_each,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c","--config", type=str, default="config/llava.yml")
    args = ap.parse_args()

    cfg = BenchConfig.from_yaml(args.config)
    ensure_dir(cfg.out_dir)
    df = pd.read_csv(cfg.csv_path)
    for col in [cfg.image_paths_column, cfg.prompt_column]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    if cfg.answer_column not in df.columns:
        df[cfg.answer_column] = ""

    model, processor = build_model_and_processor(cfg)

    run_id = f"{now_iso()}_{uuid.uuid4().hex[:8]}"
    jsonl_path = os.path.join(cfg.out_dir, f"gens_{run_id}.jsonl") if cfg.save_jsonl else None
    csv_path  = os.path.join(cfg.out_dir, f"gens_{run_id}.csv") if cfg.save_generations_csv else None
    if jsonl_path: ensure_dir(os.path.dirname(jsonl_path))
    if csv_path: ensure_dir(os.path.dirname(csv_path))

    all_rows = []
    N = len(df); bs = max(1, int(cfg.batch_size))
    for i in range(0, N, bs):
        batch = df.iloc[i:i+bs]
        convs = batch_messages(batch, cfg)
        out = generate_batch(model, processor, convs, cfg)
        for j, (_, row) in enumerate(batch.iterrows()):
            pred = out["texts"][j]; gold = row.get(cfg.answer_column, "")
            em = 1.0 if normalize_ans(pred)==normalize_ans(gold) and len(gold)>0 else 0.0
            f1 = token_f1(pred, gold) if len(gold)>0 else float("nan")
            rec = {
                "row_index": int(i+j),
                "image_paths": row[cfg.image_paths_column],
                "prompt": row[cfg.prompt_column],
                "gold": gold,
                "prediction": pred,
                "exact_match": em,
                "f1": f1,
                "preproc_ms": out["preproc_ms"],
                "generate_ms": out["generate_ms"],
                "new_tokens": out["new_tokens_each"][j],
            }
            all_rows.append(rec)
            if jsonl_path:
                with open(jsonl_path, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if csv_path:
        pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    has_gold = [r for r in all_rows if r["gold"]]
    if has_gold:
        em = sum(r["exact_match"] for r in has_gold) / len(has_gold)
        f1 = sum(0.0 if math.isnan(r["f1"]) else r["f1"] for r in has_gold) / len(has_gold)
        print(f"[Summary] N={N} | EM={em:.3f} | F1={f1:.3f}")
    if all_rows:
        avg_gen_ms = sum(r["generate_ms"] for r in all_rows) / len(all_rows)
        avg_new_toks = sum(r["new_tokens"] for r in all_rows) / len(all_rows)
        tps = (avg_new_toks / (avg_gen_ms / 1000.0)) if avg_gen_ms > 0 else float("nan")
        print(f"[Speed] avg_generate_ms={avg_gen_ms:.1f} | avg_new_tokens={avg_new_toks:.1f} | decode_toks_per_s={tps:.1f}")

if __name__ == "__main__":
    main()
