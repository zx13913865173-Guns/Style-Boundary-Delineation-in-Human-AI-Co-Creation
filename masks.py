# -*- coding: utf-8 -*-
"""
Generate task-specified target masks for the 8 style-boundary tasks.
Masks are defined by task specification, NOT by annotator segmentation.
"""

import numpy as np
from PIL import Image
from config import MASK_DIR, TASKS, IMG_SIZE


def build_mask(task: str, N: int = IMG_SIZE) -> np.ndarray:
    """Return a binary mask (uint8, 0/255) for the given task."""
    xx, yy = np.meshgrid(np.arange(N), np.arange(N))

    if task == "horizontal_split":
        m = (xx < N / 2)
    elif task == "vertical_split":
        m = (yy < N / 2)
    elif task == "diagonal_45":
        m = (yy < xx)
    elif task == "diagonal_135":
        m = (yy < (N - 1 - xx))
    elif task == "center_circle":
        cx, cy, r = N / 2, N / 2, N / 4
        m = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2)
    elif task == "corner_quadrant":
        m = (xx < N / 2) & (yy < N / 2)
    elif task == "irregular_sine":
        boundary = N / 2 + (N / 4) * np.sin(2 * np.pi * xx / N)
        m = (yy < boundary)
    elif task == "frame_border":
        t = int(N * 0.08)
        m = np.zeros((N, N), dtype=bool)
        m[:t, :] = True
        m[-t:, :] = True
        m[:, :t] = True
        m[:, -t:] = True
    else:
        raise ValueError(f"Unknown task: {task}")

    return (m.astype(np.uint8) * 255)


def generate_all_masks():
    """Save all 8 target masks to MASK_DIR."""
    for task in TASKS:
        mask = build_mask(task)
        out = MASK_DIR / f"{task}.png"
        Image.fromarray(mask).save(out)
        print(f"[masks] {task:20s} -> {out.name}")


def area_composition(task: str) -> float:
    """Return the fraction of pixels belonging to Style A (ink-wash)."""
    mask = build_mask(task)
    return float((mask == 255).mean())


if __name__ == "__main__":
    generate_all_masks()
    print("\nArea composition (Style A fraction):")
    for t in TASKS:
        print(f"  {t:20s} {area_composition(t) * 100:6.2f}%")
