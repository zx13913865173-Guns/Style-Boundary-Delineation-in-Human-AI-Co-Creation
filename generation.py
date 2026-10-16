# -*- coding: utf-8 -*-
"""
Generate images with three methods:

  - text_only   : text prompt only, no LoRA, no mask
  - global_lora : LoRA fusion at 0.5/0.5, no mask
  - rlimc       : region-wise LoRA inpainting + mask compositing
                  (hard-edge default; soft-edge variant optional)

Requires trained LoRA checkpoints in CKPT_DIR.
"""

import numpy as np
import torch
from PIL import Image, ImageFilter
from pathlib import Path
from diffusers import StableDiffusionXLInpaintPipeline

from config import (
    CKPT_DIR, GEN_DIR, MASK_DIR, TASKS,
    IMG_SIZE, SEEDS, METHODS,
)

# ============================================================
# Prompts
# ============================================================
PROMPTS = {
    "text_only": (
        "A diptych composition; the left half is a Chinese ink wash "
        "mountain landscape with misty peaks and ancient pines; "
        "the right half is a cyberpunk neon city with flying cars "
        "and holographic signs; a sharp vertical boundary divides "
        "the two halves."
    ),
    "global_lora": (
        "A landscape scene combining Chinese ink wash aesthetics "
        "and cyberpunk neon city aesthetics."
    ),
    "ink_region": (
        "Chinese ink wash mountain landscape, misty peaks, ancient pines"
    ),
    "cyber_region": (
        "cyberpunk neon city, flying cars, holographic signs, night"
    ),
}

NEG_PROMPTS = {
    "ink_region": "neon, cyberpunk, bright colors, photograph",
    "cyber_region": "ink wash, monochrome, traditional, painting",
}

GEN_PARAMS = dict(
    num_inference_steps=30,
    guidance_scale=7.5,
)


# ============================================================
# Pipeline
# ============================================================
def load_pipeline():
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.load_lora_weights(
        str(CKPT_DIR / "ink_wash" / "ink_wash.safetensors"),
        adapter_name="ink",
    )
    pipe.load_lora_weights(
        str(CKPT_DIR / "cyberpunk" / "cyberpunk.safetensors"),
        adapter_name="cyber",
    )
    return pipe


def _gen(pipe, prompt, seed, neg_prompt=None):
    return pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        generator=torch.Generator("cuda").manual_seed(seed),
        **GEN_PARAMS,
    ).images[0]


# ============================================================
# Text-Only
# ============================================================
def generate_text_only(pipe, task, seed):
    pipe.set_adapters(["ink", "cyber"], adapter_weights=[0.0, 0.0])
    img = _gen(pipe, PROMPTS["text_only"], seed)
    return img


# ============================================================
# Global LoRA Fusion
# ============================================================
def generate_global_lora(pipe, task, seed):
    pipe.set_adapters(["ink", "cyber"], adapter_weights=[0.5, 0.5])
    img = _gen(pipe, PROMPTS["global_lora"], seed)
    return img


# ============================================================
# RLIMC
# ============================================================
def generate_rlimc(pipe, task, seed, soft_edge=False, blur_radius=5):
    mask_path = MASK_DIR / f"{task}.png"
    binary_mask = Image.open(mask_path).convert("L")
    mask_np = np.array(binary_mask)

    mask_ink = (mask_np == 255).astype(np.uint8) * 255
    mask_cyber = (mask_np == 0).astype(np.uint8) * 255
    mask_ink_pil = Image.fromarray(mask_ink)
    mask_cyber_pil = Image.fromarray(mask_cyber)

    # Neutral background
    pipe.set_adapters(["ink", "cyber"], adapter_weights=[0.0, 0.0])
    bg = _gen(pipe, "a landscape scene", seed,
              neg_prompt="low quality, blurry")

    # Ink region
    pipe.set_adapters(["ink", "cyber"], adapter_weights=[1.0, 0.0])
    ink_region = pipe(
        prompt=PROMPTS["ink_region"],
        image=bg,
        mask_image=mask_ink_pil,
        negative_prompt=NEG_PROMPTS["ink_region"],
        generator=torch.Generator("cuda").manual_seed(seed),
        **GEN_PARAMS,
    ).images[0]

    # Cyber region
    pipe.set_adapters(["ink", "cyber"], adapter_weights=[0.0, 1.0])
    cyber_region = pipe(
        prompt=PROMPTS["cyber_region"],
        image=bg,
        mask_image=mask_cyber_pil,
        negative_prompt=NEG_PROMPTS["cyber_region"],
        generator=torch.Generator("cuda").manual_seed(seed),
        **GEN_PARAMS,
    ).images[0]

    # Composite
    ink_np = np.array(ink_region).astype(np.float32)
    cyber_np = np.array(cyber_region).astype(np.float32)

    if soft_edge:
        mask_soft = binary_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        alpha = np.array(mask_soft).astype(np.float32) / 255.0
    else:
        alpha = mask_np.astype(np.float32) / 255.0

    alpha = np.stack([alpha] * 3, axis=-1)
    final_np = alpha * ink_np + (1 - alpha) * cyber_np
    return Image.fromarray(final_np.astype(np.uint8))


# ============================================================
# Driver
# ============================================================
def generate_all(methods=None, seeds=None, soft_edge=False):
    methods = methods or METHODS
    seeds = seeds or SEEDS
    pipe = load_pipeline()

    for method in methods:
        out_dir = GEN_DIR / method
        out_dir.mkdir(parents=True, exist_ok=True)

        for task in TASKS:
            for seed in seeds:
                fname = out_dir / f"{task}_seed{seed}.png"
                if fname.exists():
                    continue

                if method == "text_only":
                    img = generate_text_only(pipe, task, seed)
                elif method == "global_lora":
                    img = generate_global_lora(pipe, task, seed)
                elif method == "rlimc":
                    img = generate_rlimc(pipe, task, seed,
                                         soft_edge=soft_edge)
                else:
                    raise ValueError(method)

                img.save(fname)
                print(f"[gen] {method:12s} {task:20s} seed={seed} -> {fname.name}")


if __name__ == "__main__":
    generate_all()
