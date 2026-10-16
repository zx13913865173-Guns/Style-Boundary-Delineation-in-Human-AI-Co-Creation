# -*- coding: utf-8 -*-
"""
End-to-end pipeline:

  1. Generate target masks
  2. Train LoRAs (optional, requires data)
  3. Generate images (three methods)
  4. Train style classifier
  5. Compute metrics
  6. Statistical analysis
  7. Reproduce figures
"""

import argparse
from masks import generate_all_masks
from lora_train import train_all
from generation import generate_all
from classifier import train_classifier
from metrics import compute_metrics
from statistics import run_all as run_stats
from figures import generate_all as generate_figures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-gen", action="store_true")
    parser.add_argument("--skip-clf", action="store_true")
    parser.add_argument("--only-figures", action="store_true")
    args = parser.parse_args()

    if args.only_figures:
        generate_figures()
        return

    print("=" * 60)
    print("Step 1/7: Generating target masks")
    print("=" * 60)
    generate_all_masks()

    if not args.skip_lora:
        print("=" * 60)
        print("Step 2/7: Training LoRAs")
        print("=" * 60)
        train_all()

    if not args.skip_gen:
        print("=" * 60)
        print("Step 3/7: Generating images")
        print("=" * 60)
        generate_all()

    if not args.skip_clf:
        print("=" * 60)
        print("Step 4/7: Training style classifier")
        print("=" * 60)
        train_classifier()

    print("=" * 60)
    print("Step 5/7: Computing metrics")
    print("=" * 60)
    compute_metrics()

    print("=" * 60)
    print("Step 6/7: Statistical analysis")
    print("=" * 60)
    run_stats()

    print("=" * 60)
    print("Step 7/7: Reproducing figures")
    print("=" * 60)
    generate_figures()

    print("\nAll done.")


if __name__ == "__main__":
    main()
