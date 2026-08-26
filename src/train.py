import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import FaceNasolabialDataset
from model import UNet
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Dice Loss
# ---------------------------
def dice_loss(pred, target, eps=1e-6):
    probs = F.softmax(pred, dim=1)[:, 1]
    target = target.float()
    inter = (probs * target).sum()
    union = probs.sum() + target.sum()
    return 1 - (2 * inter + eps) / (union + eps)

# ---------------------------
# IoU / Dice Metric
# ---------------------------
def iou_score(pred, target, num_classes=2):
    pred = pred.view(-1)
    target = target.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        inter = (pred_inds & target_inds).sum().item()
        union = pred_inds.sum().item() + target_inds.sum().item() - inter
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(inter / union)
    return np.nanmean(ious)

def dice_score(pred, target, eps=1e-6):
    pred = pred.view(-1)
    target = target.view(-1)
    inter = (pred * target).sum().item()
    return (2 * inter + eps) / (pred.sum().item() + target.sum().item() + eps)

# ---------------------------
# Train Loop
# ---------------------------
def main():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FaceNasolabialDataset("data", use_depth=True, resize=(256,256))
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet(in_ch=4, out_ch=2, base_ch=32).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses, ious, dices = [], [], []

    for epoch in range(50):
        model.train()
        running_loss, running_iou, running_dice = 0.0, 0.0, 0.0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            # Loss
            ce = F.cross_entropy(out, y)
            dl = dice_loss(out, y)
            loss = ce + dl

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()

            # Metrics
            pred = torch.argmax(out, dim=1)
            running_iou += iou_score(pred.cpu(), y.cpu())
            running_dice += dice_score(pred.cpu(), y.cpu())

        epoch_loss = running_loss / len(loader)
        epoch_iou = running_iou / len(loader)
        epoch_dice = running_dice / len(loader)

        losses.append(epoch_loss)
        ious.append(epoch_iou)
        dices.append(epoch_dice)

        print(f"Epoch {epoch+1:03d}: Loss={epoch_loss:.4f}, IoU={epoch_iou:.3f}, Dice={epoch_dice:.3f}")

    # 모델 저장
    torch.save(model.state_dict(), "unet_rgbd.pth")
    print("모델 저장 완료 → unet_rgbd.pth")

    # 곡선 저장
    plt.figure()
    plt.plot(losses, label="Loss")
    plt.plot(ious, label="IoU")
    plt.plot(dices, label="Dice")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title("Training Metrics")
    plt.savefig("train_metrics.png")
    print("지표 그래프 저장 완료 → train_metrics.png")

if __name__ == "__main__":
    main()
