# -*- coding: utf-8 -*-
"""
Global configuration: paths, hyperparameters, palette, task list.
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR     = ROOT / "data"
OUTPUT_DIR   = ROOT / "outputs"
FIG_DIR      = ROOT / "figures"
MASK_DIR     = OUTPUT_DIR / "target_masks"
GEN_DIR      = OUTPUT_DIR / "generated"
CKPT_DIR     = OUTPUT_DIR / "checkpoints"
METRICS_DIR  = OUTPUT_DIR / "metrics"

for d in (DATA_DIR, OUTPUT_DIR, FIG_DIR, MASK_DIR,
          GEN_DIR, CKPT_DIR, METRICS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Task list
# ============================================================
TASKS = [
    "horizontal_split",
    "vertical_split",
    "diagonal_45",
    "diagonal_135",
    "center_circle",
    "corner_quadrant",
    "irregular_sine",
    "frame_border",
]

TASK_LABELS = {
    "horizontal_split":  "Horizontal split",
    "vertical_split":    "Vertical split",
    "diagonal_45":       "Diagonal 45°",
    "diagonal_135":      "Diagonal 135°",
    "center_circle":     "Center circle",
    "corner_quadrant":   "Corner quadrant",
    "irregular_sine":    "Irregular curve",
    "frame_border":      "Frame border",
}

# ============================================================
# Image / generation
# ============================================================
IMG_SIZE = 1024
SEEDS = [42] + list(range(100, 119))  # 20 seeds total

METHODS = ["text_only", "global_lora", "rlimc"]

# ============================================================
# LoRA training (kohya-ss/sd-scripts)
# ============================================================
LORA_CONFIG = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "rank": 32,
    "alpha": 1.0,
    "steps": 3000,
    "lr": 1e-4,
    "lr_flux": 5e-5,
}

# ============================================================
# Style classifier
# ============================================================
CLASSIFIER_CONFIG = {
    "backbone": "resnet50",
    "patch_size": 32,
    "stride": 8,
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-4,
    "threshold": 0.5,
    "uncertain_lo": 0.4,
    "uncertain_hi": 0.6,
}

# ============================================================
# Statistics
# ============================================================
ALPHA = 0.05
FDR_METHOD = "fdr_bh"

# ============================================================
# Color palette (SCI blue academic)
# ============================================================
BLUE_DARK   = "#0B3C6B"
BLUE_MID    = "#1F6FB2"
BLUE_LIGHT  = "#7FB3D5"
BLUE_PALE   = "#C9DDF0"
BLUE_TRANS  = "#4A90C2"
BLUE_ACCENT = "#2E86C1"
BLUE_GRAY   = "#5D6D7E"

FIG_SIZE = (16, 9)
DPI = 900
