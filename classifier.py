# -*- coding: utf-8 -*-
"""
Style classifier: ResNet-50 sliding-window patch classifier.

Used for:
  - patch-level style-region segmentation (boundary alignment)
  - transition-zone width estimation
  - style-purity sanity check
"""

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

from config import (
    DATA_DIR, CKPT_DIR, METRICS_DIR, CLASSIFIER_CONFIG,
)

STYLES = ["ink_wash", "cyberpunk"]
STYLE_TO_IDX = {s: i for i, s in enumerate(STYLES)}


# ============================================================
# Dataset
# ============================================================
class StyleDataset(Dataset):
    def __init__(self, root: Path, transform=None):
        self.samples = []
        for style in STYLES:
            for p in (root / style).glob("*.jpg"):
                self.samples.append((p, STYLE_TO_IDX[style]))
            for p in (root / style).glob("*.png"):
                self.samples.append((p, STYLE_TO_IDX[style]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# Model
# ============================================================
def build_classifier(num_classes=2):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
# Train
# ============================================================
def train_classifier(data_root: Path = None,
                     out_ckpt: Path = None):
    data_root = data_root or (DATA_DIR / "classifier")
    out_ckpt = out_ckpt or (CKPT_DIR / "classifier.pth")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    ds = StyleDataset(data_root, transform=tf)
    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=CLASSIFIER_CONFIG["batch_size"],
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=CLASSIFIER_CONFIG["batch_size"],
                            shuffle=False, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_classifier().to(device)

    opt = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_CONFIG["lr"])
    crit = nn.CrossEntropyLoss()

    for epoch in range(CLASSIFIER_CONFIG["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(total, 1)
        print(f"[classifier] epoch {epoch + 1:3d}  val_acc={acc:.4f}")

    torch.save(model.state_dict(), out_ckpt)
    print(f"[classifier] saved -> {out_ckpt}")
    return model


# ============================================================
# Inference: patch-level probability map
# ============================================================
@torch.no_grad()
def predict_prob_map(model, image: Image.Image, device="cuda"):
    """
    Returns:
        prob_map: HxW float32, probability of Style B (cyberpunk)
        uncertain_mask: HxW bool
    """
    model.eval()
    ps = CLASSIFIER_CONFIG["patch_size"]
    stride = CLASSIFIER_CONFIG["stride"]

    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    W, H = image.size
    prob_map = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    patches = []
    coords = []
    for y in range(0, H - ps + 1, stride):
        for x in range(0, W - ps + 1, stride):
            patch = image.crop((x, y, x + ps, y + ps))
            patches.append(tf(patch))
            coords.append((x, y))

    if not patches:
        return prob_map, np.zeros((H, W), dtype=bool)

    batch_size = 64
    for i in range(0, len(patches), batch_size):
        batch = torch.stack(patches[i:i + batch_size]).to(device)
        probs = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
        for (x, y), p in zip(coords[i:i + batch_size], probs):
            prob_map[y:y + ps, x:x + ps] += p
            count_map[y:y + ps, x:x + ps] += 1

    prob_map = prob_map / np.maximum(count_map, 1)
    lo, hi = CLASSIFIER_CONFIG["uncertain_lo"], CLASSIFIER_CONFIG["uncertain_hi"]
    uncertain = (prob_map >= lo) & (prob_map <= hi)
    return prob_map, uncertain


def load_classifier(ckpt: Path = None):
    ckpt = ckpt or (CKPT_DIR / "classifier.pth")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_classifier().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    return model


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.train:
        train_classifier()
