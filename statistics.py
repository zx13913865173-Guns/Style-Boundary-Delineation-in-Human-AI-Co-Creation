# -*- coding: utf-8 -*-
"""
Statistical analysis:

  - Linear mixed-effects models (Model A / B / C)
  - 2x2 factorial ANOVA on ablation
  - FDR correction
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

from config import METRICS_DIR, ALPHA, FDR_METHOD


# ============================================================
# Load
# ============================================================
def load_metrics():
    return pd.read_csv(METRICS_DIR / "objective_metrics.csv")


# ============================================================
# Model A: three-method comparison
# ============================================================
def model_a(df: pd.DataFrame, dv="iou"):
    formula = f"{dv} ~ C(method)"
    md = smf.mixedlm(formula, df, groups=df["task"])
    mdf = md.fit(reml=True)
    print(f"\n[Model A] DV = {dv}")
    print(mdf.summary())
    return mdf


# ============================================================
# Model B: 2x2 factorial ablation
# ============================================================
def build_factorial(df: pd.DataFrame) -> pd.DataFrame:
    """Map three-method labels to 2x2 LoRA/Mask factors."""
    mapping = {
        "rlimc":       dict(lora=1, mask=1),
        "global_lora": dict(lora=1, mask=0),
        "text_only":   dict(lora=0, mask=0),
    }
    # Mask-only condition should be included if generated separately
    rows = []
    for _, r in df.iterrows():
        if r["method"] not in mapping:
            continue
        f = mapping[r["method"]]
        rows.append(dict(
            task=r["task"], seed=r["seed"],
            lora=f["lora"], mask=f["mask"],
            iou=r["iou"],
        ))
    return pd.DataFrame(rows)


def model_b(df_fac: pd.DataFrame, dv="iou"):
    formula = f"{dv} ~ lora * mask"
    md = smf.mixedlm(formula, df_fac, groups=df_fac["task"])
    mdf = md.fit(reml=True)
    print(f"\n[Model B] 2x2 factorial, DV = {dv}")
    print(mdf.summary())
    return mdf


# ============================================================
# Model C: generalization
# ============================================================
def model_c(df: pd.DataFrame, dv="iou"):
    formula = f"{dv} ~ C(model) * C(method)"
    md = smf.mixedlm(formula, df, groups=df["task"])
    mdf = md.fit(reml=True)
    print(f"\n[Model C] generalization, DV = {dv}")
    print(mdf.summary())
    return mdf


# ============================================================
# FDR-corrected pairwise comparisons
# ============================================================
def pairwise_fdr(df: pd.DataFrame, group="method", dv="iou"):
    methods = df[group].unique()
    pvals, pairs = [], []
    for i, a in enumerate(methods):
        for b in methods[i + 1:]:
            xa = df[df[group] == a][dv].dropna()
            xb = df[df[group] == b][dv].dropna()
            t, p = sm.stats.ttest_ind(xa, xb, usevar="unequal")
            pvals.append(p)
            pairs.append((a, b))

    reject, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method=FDR_METHOD)

    print(f"\n[FDR] pairwise {group} on {dv}")
    for (a, b), p, pa, rj in zip(pairs, pvals, p_adj, reject):
        print(f"  {a:20s} vs {b:20s}  p={p:.4f}  p_adj={pa:.4f}  sig={rj}")
    return pd.DataFrame(dict(pair=pairs, p=pvals, p_adj=p_adj, reject=reject))


# ============================================================
# Main
# ============================================================
def run_all():
    df = load_metrics()
    model_a(df, "iou")
    model_a(df, "transition_width")
    model_a(df, "style_purity_A")
    model_a(df, "style_purity_B")

    fac = build_factorial(df)
    if not fac.empty:
        model_b(fac, "iou")

    pairwise_fdr(df, "method", "iou")
    pairwise_fdr(df, "method", "transition_width")


if __name__ == "__main__":
    run_all()
