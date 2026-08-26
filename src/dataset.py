import os, json, cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class FaceNasolabialDataset(Dataset):
    def __init__(self, root, use_depth=True, resize=(256, 256)):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "masks")
        self.depth_dir = os.path.join(root, "depth")
        self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.img_dir)
                    if f.endswith((".png", ".jpg", ".jpeg"))]
        self.use_depth = use_depth
        self.resize = resize

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]

        # ---- RGB ----
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = os.path.join(self.img_dir, id_+ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        img = np.array(Image.open(img_path).convert("RGB"))

        # ---- Mask ----
        mask_path = os.path.join(self.mask_dir, id_+"_mask.png")
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 0).astype(np.int64)

        # ---- Depth ----
        if self.use_depth:
            dfile = os.path.join(self.depth_dir, id_+".json")
            with open(dfile) as f:
                dj = json.load(f)
            w, h = dj["Width"], dj["Height"]
            depth = np.array(dj["Depth"], dtype=np.float32).reshape(h, w)
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            depth = (depth - np.nanmean(depth)) / (np.nanstd(depth)+1e-6)
            img = np.concatenate([img.astype(np.float32)/255.0, depth[...,None]], axis=-1)
        else:
            img = img.astype(np.float32)/255.0

        # ---- Resize ----
        if self.resize:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)

        # ---- Tensor 변환 ----
        img = torch.from_numpy(img).permute(2,0,1).float()
        mask = torch.from_numpy(mask).long()
        return img, mask
