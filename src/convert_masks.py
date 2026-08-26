import os
import json
import numpy as np
from PIL import Image, ImageDraw

# 입력 폴더: labelme json이 들어 있는 곳
LABELME_JSON_DIR = "data/labels_json"
# 출력 폴더: 마스크 png 저장 위치
MASK_DIR = "data/masks"

os.makedirs(MASK_DIR, exist_ok=True)

def labelme_json_to_mask(json_path, out_path):
    import json
    from PIL import Image, ImageDraw

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h, w = data["imageHeight"], data["imageWidth"]
    mask = Image.new("L", (w, h), 0)

    for shape in data["shapes"]:
        label = shape["label"].strip().lower()
        if label == "wrinkle":
            pts = [tuple(map(int, p)) for p in shape["points"]]
            ImageDraw.Draw(mask).polygon(pts, outline=255, fill=255)

    mask.save(out_path)


def main():
    for fname in os.listdir(LABELME_JSON_DIR):
        if not fname.endswith(".json"):
            continue
        json_path = os.path.join(LABELME_JSON_DIR, fname)
        out_name = fname.replace(".json", "_mask.png")
        out_path = os.path.join(MASK_DIR, out_name)
        labelme_json_to_mask(json_path, out_path)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
