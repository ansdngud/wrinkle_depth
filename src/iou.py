'''
import numpy as np
from PIL import Image

def load_mask(path, size=(512, 512)):
    """흑백 마스크 이미지를 0과 1로 변환하고 size로 리사이즈"""
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, Image.NEAREST)  # GT 마스크는 NEAREST로 리사이즈
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8)
    return mask

def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return 2.0 * intersection / (y_true.sum() + y_pred.sum() + 1e-8)

if __name__ == "__main__":
    gt_mask   = load_mask("/mnt/nas4/mwh/depth2/data/masks/image10_mask.png", size=(512, 512))     # 정답 마스크
    pred_mask = load_mask("/mnt/nas4/mwh/depth2/results_rgbd/image10_mask.png", size=(512, 512))   # 예측 마스크

    iou  = iou_score(gt_mask, pred_mask)
    dice = dice_score(gt_mask, pred_mask)

    print("IoU :", iou)
    print("Dice:", dice)
'''

import numpy as np
from PIL import Image

def load_mask(path, size=(512, 512)):
    """흑백 마스크 이미지를 0과 1로 변환하고 size로 리사이즈"""
    mask = Image.open(path).convert("L")
    mask = mask.resize(size, Image.NEAREST)  # GT 마스크는 NEAREST로 리사이즈
    mask = np.array(mask)
    mask = (mask > 127).astype(np.uint8)
    return mask

def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred).sum()
    return 2.0 * intersection / (y_true.sum() + y_pred.sum() + 1e-8)

def precision_score(y_true, y_pred):
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score(y_true, y_pred):
    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

if __name__ == "__main__":
    gt_mask   = load_mask("/mnt/nas4/mwh/depth2/data/masks/image10_mask.png", size=(512, 512))     # 정답 마스크
    pred_mask = load_mask("/mnt/nas4/mwh/depth2/results_rgbd/image10_mask.png", size=(512, 512))   # 예측 마스크

    iou  = iou_score(gt_mask, pred_mask)
    dice = dice_score(gt_mask, pred_mask)
    prec = precision_score(gt_mask, pred_mask)
    rec  = recall_score(gt_mask, pred_mask)

    print("IoU       :", iou)
    print("Dice      :", dice)
    print("Precision :", prec)
    print("Recall    :", rec)
