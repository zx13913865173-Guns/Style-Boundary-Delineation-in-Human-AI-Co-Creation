# -*- coding: utf-8 -*-
"""
Objective metrics:

  - boundary alignment IoU (against task-specified target masks)
  - mean boundary distance
  - transition zone width (0.1–0.9 probability distance)
  - style purity (CLIP similarity)
"""

import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from pathlib import Path
from scipy.ndimage import distance_transform_edt

from config import (
    MASK_DIR, GEN_DIR, DATA_DIR, METRICS_DIR,
    TASKS, METHODS, SEEDS, IMG_SIZE,
)
from classifier import load_classifier, predict_prob_map


# ============================================================
# Boundary alignment IoU
# ============================================================
def boundary_iou(pred_mask: np.ndarray, target_mask: np.ndarray,
                 uncertain: np.ndarray = None) -> float:
    """
    pred_mask, target_mask: HxW bool
    uncertain: HxW bool, excluded from IoU if provided
    """
    if uncertain is not None:
        valid = ~uncertain
    else:
        valid = np.ones_like(pred_mask, dtype=bool)

    inter = np.logical_and(pred_mask, target_mask) & valid
    union = np.logical_or(pred_mask, target_mask) & valid
    if union.sum() == 0:
        return float("nan")
    return float(inter.sum() / union.sum())


def mean_boundary_distance(pred_mask: np.ndarray,
                           target_mask: np.ndarray) -> float:
    """Mean distance (px) from predicted boundary to target boundary."""
    pred_edge = _edge(pred_mask)
    target_edge = _edge(target_mask)
    if pred_edge.sum() == 0 or target_edge.sum() == 0:
        return float("nan")
    dt = distance_transform_edt(~target_edge)
    return float(dt[pred_edge].mean())


def _edge(mask: np.ndarray) -> np.ndarray:
    er = mask & ~np.roll(mask, 1, 0) & ~np.roll(mask, -1, 0) \
              & ~np.roll(mask, 1, 1) & ~np.roll(mask, -1, 1)
    return er


# ============================================================
# Transition zone width
# ============================================================
def transition_width(prob_map: np.ndarray) -> float:
    """
    Width in pixels over which style probability changes from 0.1 to 0.9.
    Computed along horizontal scanlines and averaged.
    """
    widths = []
    H, W = prob_map.shape
    for y in range(H):
        row = prob_map[y]
        lo_idx = np.where(row >= 0.1)[0]
        hi_idx = np.where(row >= 0.9)[0]
        if len(lo_idx) == 0 or len(hi_idx) == 0:
            continue
        w = abs(hi_idx[0] - lo_idx[0])
        if w > 0:
            widths.append(w)
    return float(np.mean(widths)) if widths else float("nan")


# ============================================================
# Style purity (CLIP)
# ============================================================
class CLIPPurity:
    def __init__(self, ref_dir: Path = None, device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        ref_dir = ref_dir or (DATA_DIR / "style_reference")
        self.ref_emb = {}
        for style in ["ink_wash", "cyberpunk"]:
            embs = []
            for p in (ref_dir / style).glob("*.jpg"):
                img = self.preprocess(Image.open(p)).unsqueeze(0).to(device)
                with torch.no_grad():
                    embs.append(self.model.encode_image(img))
            if embs:
                self.ref_emb[style] = torch.cat(embs).mean(0, keepdim=True)
                self.ref_emb[style] /= self.ref_emb[style].norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def score(self, region: Image.Image, style: str) -> float:
        if style not in self.ref_emb:
            return float("nan")
        img = self.preprocess(region).unsqueeze(0).to(self.device)
        emb = self.model.encode_image(img)
        emb /= emb.norm(dim=-1, keepdim=True)
        return float((emb @ self.ref_emb[style].T).item())


# ============================================================
# Main
# ============================================================
def compute_metrics(methods=None, seeds=None):
    methods = methods or METHODS
    seeds = seeds or SEEDS

    clf = load_classifier()
    clip_metric = CLIPPurity()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for method in methods:
        for task in TASKS:
            target = np.array(Image.open(MASK_DIR / f"{task}.png")) == 255
            for seed in seeds:
                fname = GEN_DIR / method / f"{task}_seed{seed}.png"
                if not fname.exists():
                    continue

                img = Image.open(fname).convert("RGB")
                prob_map, uncertain = predict_prob_map(clf, img, device)

                pred_mask = prob_map >= 0.5
                iou = boundary_iou(pred_mask, target, uncertain)
                bdist = mean_boundary_distance(pred_mask, target)
                tw = transition_width(prob_map)

                # Style purity: crop region A and B
                mask_np = np.array(Image.open(MASK_DIR / f"{task}.png"))
                regionA = _crop_region(img, mask_np == 255)
                regionB = _crop_region(img, mask_np == 0)
                purityA = clip_metric.score(regionA, "ink_wash")
                purityB = clip_metric.score(regionB, "cyberpunk")

                rows.append(dict(
                    method=method, task=task, seed=seed,
                    iou=iou, boundary_dist=bdist,
                    transition_width=tw,
                    style_purity_A=purityA,
                    style_purity_B=purityB,
                    uncertain_ratio=float(uncertain.mean()),
                ))
                print(f"[metrics] {method:12s} {task:20s} seed={seed} "
                      f"IoU={iou:.3f} tw={tw:.1f}")

    df = pd.DataFrame(rows)
    out = METRICS_DIR / "objective_metrics.csv"
    df.to_csv(out, index=False)
    print(f"[metrics] saved -> {out}")
    return df


def _crop_region(img: Image.Image, mask: np.ndarray) -> Image.Image:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return img
    return img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


if __name__ == "__main__":
    compute_metrics()
