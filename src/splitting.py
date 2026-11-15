"""Reproducible 70:15:15 stratified split + leakage audit.

Strategy (documented limitation): per-class seeded random split on the ORIGINAL label. Flow records from the same
capture session are not independent, so random stratification can give optimistic test scores - especially for
CICIDS2017/InSDN (consecutive flows of one attack run). `leakage_audit` quantifies cross-split near-duplicates and
timestamp overlap so this limitation can be reported with numbers.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def stratified_split_indices(y, seed: int, fractions=(0.70, 0.15, 0.15)) -> tuple[dict, dict]:
    """Returns ({'train','validation','test': positional index arrays}, metadata). Classes too small for all three
    splits are handled deterministically: n>=3 -> >=1 per split; n==2 -> train+test; n==1 -> train only."""
    y = np.asarray(y)
    f_tr, f_va, f_te = fractions
    assert abs(f_tr + f_va + f_te - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    parts = {"train": [], "validation": [], "test": []}
    tiny = []
    for cls in sorted(pd.unique(y).tolist(), key=str):
        idx = np.flatnonzero(y == cls)
        idx = idx[rng.permutation(len(idx))]
        n = len(idx)
        if n >= 3:
            n_va = max(1, int(round(f_va * n))); n_te = max(1, int(round(f_te * n)))
            n_tr = n - n_va - n_te
            if n_tr < 1:
                n_tr, n_va, n_te = n - 2, 1, 1
        elif n == 2:
            n_tr, n_va, n_te = 1, 0, 1
        else:
            n_tr, n_va, n_te = n, 0, 0
        if n < 7:  # cannot honour 70/15/15 within +-1 sample
            tiny.append(dict(label=str(cls), n=n, train=n_tr, validation=n_va, test=n_te))
        parts["train"].append(idx[:n_tr]); parts["validation"].append(idx[n_tr:n_tr + n_va]); parts["test"].append(idx[n_tr + n_va:])
    out = {k: np.sort(np.concatenate(v)) if v else np.array([], dtype=int) for k, v in parts.items()}
    n = len(y)
    meta = dict(strategy="per-class seeded random split on label_original (manual stratification)", seed=seed,
                requested_fractions=dict(train=f_tr, validation=f_va, test=f_te),
                achieved_fractions={k: len(v) / n for k, v in out.items()}, sizes={k: int(len(v)) for k, v in out.items()},
                classes_too_small_for_exact_stratification=tiny)
    return out, meta

def split_class_table(labels: pd.Series, idx: dict) -> pd.DataFrame:
    rows = {k: labels.iloc[v].value_counts() for k, v in idx.items()}
    t = pd.DataFrame(rows).fillna(0).astype(int)
    t["total"] = t.sum(axis=1)
    return t.sort_values("total", ascending=False)

def leakage_audit(df: pd.DataFrame, idx: dict, feature_cols: list[str], decimals: int = 3) -> dict:
    """Cross-split exact/near-duplicate and temporal-overlap audit (read-only; never modifies the split)."""
    sub = df[feature_cols].copy()
    for c in feature_cols:
        if pd.api.types.is_float_dtype(sub[c]):
            sub[c] = sub[c].round(decimals)
    h = pd.util.hash_pandas_object(sub, index=False).to_numpy()
    hx = pd.util.hash_pandas_object(df[feature_cols], index=False).to_numpy()
    res = {"near_duplicate_rounding_decimals": decimals}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        res[f"near_duplicate_rows_{b}_in_{a}"] = int(np.isin(h[idx[b]], h[idx[a]]).sum())
        res[f"exact_duplicate_rows_{b}_in_{a}"] = int(np.isin(hx[idx[b]], hx[idx[a]]).sum())
        res[f"near_duplicate_pct_of_{b}_in_{a}"] = 100 * res[f"near_duplicate_rows_{b}_in_{a}"] / max(1, len(idx[b]))
    if "__timestamp" in df.columns and df["__timestamp"].notna().any():
        res["timestamp_ranges"] = {k: [str(df["__timestamp"].iloc[v].min()), str(df["__timestamp"].iloc[v].max())] for k, v in idx.items()}
        res["temporal_note"] = "Splits are interleaved in time (random split); ranges overlap by construction."
    else:
        res["temporal_note"] = "No timestamp available for this dataset; temporal leakage cannot be quantified."
    return res
