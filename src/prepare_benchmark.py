import argparse
import json
import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

# Configuration
JSON_PATH = "data/annotations.jsonl"
IMAGE_ROOT = "data/raw_images/test2017"
CROP_DIR = "data/crops"
OUT_CSV_SINGLE = "data/bench_single.csv"
OUT_CSV_MULTI = "data/bench_multi.csv"

def normalize_bbox(bbox, width, height):
    """
    Normalize coordinates to [0.0, 1.0] rounded to 2 decimals.
    """
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / width, 2),
        round(y1 / height, 2),
        round(x2 / width, 2),
        round(y2 / height, 2)
    ]

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare benchmark CSVs and crops")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples (rows) to generate.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer")
    return args


def main(limit=None):
    os.makedirs(CROP_DIR, exist_ok=True)

    single_rows = []
    multi_rows = []
    
    with open(JSON_PATH, 'r') as f:
        lines = f.readlines()

    count = 0
    
    # We use a progress bar, but we might break early
    pbar = tqdm(total=limit if limit else len(lines))
    
    for line in lines:
        # 1. STRICT LIMIT CHECK (Based on rows/count, not images)
        if limit is not None and count >= limit:
            break
            
        if not line.strip(): continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        filename = entry["filename"]
        img_w = entry["width"]
        img_h = entry["height"]
        
        src_img_path = os.path.join(IMAGE_ROOT, filename)
        if not os.path.exists(src_img_path):
            continue
            
        try:
            with Image.open(src_img_path) as im:
                regions = entry.get("grounding", {}).get("regions", [])
                
                for i, region in enumerate(regions):
                    # Double check limit inside the loop in case an image has many regions
                    if limit is not None and count >= limit:
                        break

                    bbox = region["bbox"] # [x1, y1, x2, y2]
                    phrase = region["phrase"]
                    attributes = ", ".join(region.get("attributes", []))
                    
                    # Construct Ground Truth
                    # Labeling Note: This creates a "Bag of Words" style GT.
                    # Good for recall-based evaluation.
                    gt_answer = f"{phrase}"
                    if attributes:
                        gt_answer += f" with attributes: {attributes}"

                    # --- 1. PREPARE CROPS ---
                    x1, y1, x2, y2 = bbox
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(img_w, x2); y2 = min(img_h, y2)
                    
                    if x2 - x1 < 5 or y2 - y1 < 5:
                        continue

                    crop = im.crop((x1, y1, x2, y2))
                    crop_name = f"{filename.split('.')[0]}_crop_{i}.jpg"
                    crop_path = os.path.join(CROP_DIR, crop_name)
                    crop.save(crop_path)

                    # --- 2. SINGLE IMAGE ROW ---
                    # Input: Global Only
                    # Context: Normalized BBox coordinates
                    norm_box = normalize_bbox([x1, y1, x2, y2], img_w, img_h)
                    
                    # Note: We use {bbox} placeholder if your data_loading supports it, 
                    # otherwise we bake the values in here.
                    single_prompt = (
                        f"Identify the object located at coordinates {norm_box} "
                        f"(x1, y1, x2, y2) relative to the image size. "
                        f"Describe it and list its attributes."
                    )
                    
                    single_rows.append({
                        "image_paths": src_img_path,
                        "prompt": single_prompt,
                        "answer": gt_answer
                    })

                    # --- 3. MULTI IMAGE ROW ---
                    # Input Order: Global;Crop
                    multi_paths = f"{src_img_path};{crop_path}"
                    
                    # Prompt: Global is first, Crop is second
                    multi_prompt = (
                        "The first image is the global view. The second image is a cropped detail. "
                        "Identify the object shown in the cropped detail and list its attributes."
                    )
                    
                    multi_rows.append({
                        "image_paths": multi_paths,
                        "prompt": multi_prompt,
                        "answer": gt_answer
                    })

                    count += 1
                    pbar.update(1)
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    pbar.close()

    # Save CSVs
    pd.DataFrame(single_rows).to_csv(OUT_CSV_SINGLE, index=False)
    pd.DataFrame(multi_rows).to_csv(OUT_CSV_MULTI, index=False)
    
    print(f"\nProcessing complete.")
    print(f"Total samples generated: {count}")
    print(f"Single-image benchmark: {OUT_CSV_SINGLE}")
    print(f"Multi-image benchmark:  {OUT_CSV_MULTI}")

if __name__ == "__main__":
    args = parse_args()
    main(limit=args.limit)