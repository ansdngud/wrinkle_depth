import torch
import numpy as np
import cv2
import json
from PIL import Image
from model import UNet   # 학습에 사용한 UNet (in_ch=4, out_ch=2, base_ch=32)

def load_model(model_path, device):
    model = UNet(in_ch=4, out_ch=2, base_ch=32).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_rgbd(image_path, depth_path, resize=(256,256)):
    # RGB
    rgb = np.array(Image.open(image_path).convert("RGB"))

    # Depth
    with open(depth_path) as f:
        dj = json.load(f)
    w, h = dj["Width"], dj["Height"]
    depth = np.array(dj["Depth"], dtype=np.float32).reshape(h, w)
    depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    depth = (depth - np.nanmean(depth)) / (np.nanstd(depth) + 1e-6)

    # 합치기 (RGB+Depth)
    img = np.concatenate([rgb.astype(np.float32)/255.0, depth[...,None]], axis=-1)

    # 학습과 동일하게 resize
    img_resized = cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)

    # Tensor 변환
    tensor = torch.from_numpy(img_resized).permute(2,0,1).unsqueeze(0).float()
    return tensor, rgb

def inference_single(model, image_path, depth_path, save_dir="results", device="cuda:0", alpha=0.5):
    import os
    os.makedirs(save_dir, exist_ok=True)

    # 전처리
    x, rgb = preprocess_rgbd(image_path, depth_path, resize=(256,256))
    x = x.to(device)

    # 추론
    with torch.no_grad():
        out = model(x)
        pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy()

    # 원본 크기로 되돌리기
    pred_resized = cv2.resize(pred.astype(np.uint8), (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Overlay
    color_mask = np.zeros_like(rgb)
    color_mask[pred_resized == 1] = [255,0,0]
    blended = cv2.addWeighted(rgb, 1-alpha, color_mask, alpha, 0)

    # 저장
    base = os.path.splitext(os.path.basename(image_path))[0]
    mask_path = os.path.join(save_dir, f"{base}_mask.png")
    overlay_path = os.path.join(save_dir, f"{base}_overlay.png")
    cv2.imwrite(mask_path, pred_resized*255)
    cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    print(f"✅ Saved: {mask_path}, {overlay_path}")

if __name__ == "__main__":
    # 입력 파일
    image_path = "/mnt/nas4/mwh/depth2/new_data/images/image10.png"
    depth_path = "/mnt/nas4/mwh/depth2/new_data/depth/image10.json"
    model_path = "unet_rgbd.pth"

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)

    inference_single(model, image_path, depth_path, save_dir="results_rgbd", device=device)
