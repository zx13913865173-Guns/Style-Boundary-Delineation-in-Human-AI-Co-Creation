# -*- coding: utf-8 -*-
"""
LoRA training wrapper for kohya-ss/sd-scripts.

Expects training images in DATA_DIR/{style}/ with captions.
Uses the hyperparameters from config.LORA_CONFIG.
"""

import subprocess
from pathlib import Path
from config import DATA_DIR, CKPT_DIR, LORA_CONFIG

STYLES = ["ink_wash", "cyberpunk"]

KOHYA_TRAIN = "accelerate launch sd-scripts/train_network.py"


def train_lora(style: str, base_model: str = None, lr: float = None):
    """Train one LoRA for the given style."""
    base_model = base_model or LORA_CONFIG["base_model"]
    lr = lr or LORA_CONFIG["lr"]

    train_dir = DATA_DIR / style
    out_dir = CKPT_DIR / style
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        *KOHYA_TRAIN.split(),
        f"--pretrained_model_name_or_path={base_model}",
        f"--train_data_dir={train_dir}",
        f"--output_dir={out_dir}",
        f"--output_name={style}",
        "--network_module=networks.lora",
        f"--network_dim={LORA_CONFIG['rank']}",
        f"--network_alpha={LORA_CONFIG['alpha']}",
        f"--max_train_steps={LORA_CONFIG['steps']}",
        f"--learning_rate={lr}",
        "--resolution=1024,1024",
        "--mixed_precision=fp16",
        "--save_model_as=safetensors",
        "--optimizer_type=AdamW8bit",
        "--lr_scheduler=cosine",
        "--seed=42",
    ]

    print(f"[lora_train] Training {style} ...")
    subprocess.run(cmd, check=True)
    print(f"[lora_train] Done: {out_dir / (style + '.safetensors')}")


def train_all(base_model: str = None, lr: float = None):
    for style in STYLES:
        train_lora(style, base_model=base_model, lr=lr)


if __name__ == "__main__":
    train_all()
