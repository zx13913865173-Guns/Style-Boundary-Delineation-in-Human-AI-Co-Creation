# Style Boundary Delineation in Human–AI Co-Creation

Reproducible materials for the paper:

> **Style Boundary Delineation in Human–AI Co-Creation: Evaluating Region-wise LoRA Inpainting with Mask Compositing for Multi-Style Control**
> Xin Zhang, Dmitry Galkin
> Tomsk State University

## Overview

This repository contains:

- Target-mask generation for 8 style-boundary tasks
- LoRA training pipeline (Chinese ink-wash, cyberpunk)
- Three generation methods: Text-Only, Global LoRA Fusion, RLIMC
- Style classifier (ResNet-50 sliding-window patch classifier)
- Objective metrics: boundary alignment IoU, transition width, style purity
- Subjective naturalness aggregation
- Statistical models: linear mixed-effects models
- Figures 1–6 (16:9, 900 DPI, SCI blue palette)

## Hardware & Environment

| Item | Specification |
|---|---|
| GPU | NVIDIA RTX 4070 Super 12GB |
| CUDA | 12.1 |
| OS | Windows 11 Pro 23H2 |
| Python | 3.10.11 |
| PyTorch | 2.1.0+cu121 |
| Diffusers | 0.25.0 |
| xformers | 0.0.23 |

## Quick Start

```bash
pip install -r requirements.txt

# 1. Generate target masks for 8 tasks
python masks.py

# 2. Train LoRAs (requires training data)
python lora_train.py

# 3. Generate images with three methods
python generation.py

# 4. Train style classifier
python classifier.py --train

# 5. Compute metrics
python metrics.py

# 6. Statistical analysis
python statistics.py

# 7. Reproduce figures
python figures.py
