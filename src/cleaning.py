"""Documented, label-aware cleaning. Every removal is logged with a reason; nothing is silently dropped.

Leakage note: cleaning never imputes. Missing values are only converted to NaN here and imputed later by a
pipeline fitted on the training split. Constant-over-the-whole-dataset columns are constant in train as well,
so removing them cannot leak test information.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .label_mapping import add_label_columns, normalize_label
from .utils import feature_columns, conflicting_duplicate_groups

def _col_hash(s: pd.Series) -> int:
    return int(pd.util.hash_pandas_object(s, index=False).sum())

def clean_dataset(name: str, raw: pd.DataFrame, cfg: dict, copy: bool = True) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Returns (cleaned_df, report, removed_columns_df)."""
    cc = cfg["cleaning"]; ds = cfg["datasets"][name]
    cat_cols = ds["categorical_columns"]
    df = raw.copy() if copy else raw      # copy=False halves peak memory on CICIDS2017 (caller must not reuse `raw`)
    removed: list[dict] = []
    before = dict(rows=len(df), columns=int(len([c for c in df.columns if not c.startswith("__")])))
    lab_raw = df[ds["label_column"]]
    lab_ok = lab_raw.dropna()
    lab_ok = lab_ok[lab_ok.astype(str).str.strip() != ""]
    before["class_counts"] = lab_ok.map(lambda x: normalize_label(x, name)).value_counts().to_dict()   # labelled rows only
    before["rows_without_label"] = int(len(df) - len(lab_ok))

    # 1. identifier / non-traffic columns (explicit config list only)
    for col, reason in cfg["identifier_columns"].items():
        if col in df.columns:
            if col == "Timestamp":
                df["__timestamp"] = pd.to_datetime(df[col], errors="coerce", dayfirst=bool(ds.get("timestamp_dayfirst", False)), format="mixed")
            if col == "difficulty":
                df["__difficulty"] = df[col]
            df = df.drop(columns=[col])
            removed.append(dict(column=col, reason=reason, stage="identifier"))

    label_col = ds["label_column"]
    protected = {label_col} | {c for c in df.columns if c.startswith("__")}
    feats = [c for c in df.columns if c not in protected]

    # 2. numeric coercion + infinities -> NaN (counted)
    num_cols = [c for c in feats if c not in cat_cols]
    n_coerced = n_inf = 0
    for c in num_cols:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            t = pd.to_numeric(s, errors="coerce"); n_coerced += int(t.isna().sum() - s.isna().sum()); s = t
        arr = s.to_numpy(dtype="float64", na_value=np.nan)
        m = np.isinf(arr)
        if m.any():
            n_inf += int(m.sum()); arr = np.where(m, np.nan, arr)
        df[c] = arr
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
            df[c] = df[c].map(lambda v: v if (isinstance(v, float) and np.isnan(v)) else str(v).strip())

    # 3. rows without label / completely empty rows
    n0 = len(df)
    no_label = df[label_col].isna() | (df[label_col].astype(str).str.strip() == "")
    df = df[~no_label]
    n_no_label = n0 - len(df)
    n_all_nan = 0
    if cc["drop_all_nan_rows"]:
        empty = df[feats].isna().all(axis=1)
        n_all_nan = int(empty.sum()); df = df[~empty]

    # 4. all-NaN / constant / duplicate-content columns (computed on the whole raw data; see module docstring)
    if cc["drop_constant_columns"]:
        for c in feats:
            if c in df.columns and df[c].nunique(dropna=True) <= 1:
                removed.append(dict(column=c, reason="constant (<=1 distinct non-null value) over the whole dataset", stage="constant"))
                df = df.drop(columns=[c])
    if cc["drop_duplicate_columns"]:
        seen: dict[int, str] = {}
        for c in [c for c in feats if c in df.columns]:
            h = _col_hash(df[c])
            if h in seen and df[c].equals(df[seen[h]]):
                removed.append(dict(column=c, reason=f"values identical to column '{seen[h]}'", stage="duplicate_column"))
                df = df.drop(columns=[c])
            else:
                seen[h] = c

    # 5. labels (original text-normalised, harmonised, binary)
    df = add_label_columns(df, name, label_col)

    # 6. exact duplicate rows: all features + original label, keep first occurrence in file order
    fcols = feature_columns(df)
    n1 = len(df)
    dup_removed_by_class = {}
    if cc["drop_exact_duplicates"]:
        dup_mask = df.duplicated(subset=fcols + ["label_original"], keep="first")
        dup_removed_by_class = df.loc[dup_mask, "label_original"].value_counts().to_dict()
        df = df[~dup_mask]
    n_dup = n1 - len(df)
    # report (not remove) conflicting duplicates
    n_conflict_groups = conflicting_duplicate_groups(df, fcols, df["label_original"])

    df = df.reset_index(drop=True)
    after = dict(rows=len(df), columns=len(fcols), class_counts=df["label_original"].value_counts().to_dict())
    report = dict(dataset=name, before=before, after=after,
                  rows_removed=dict(no_label=n_no_label, all_features_nan=n_all_nan, exact_duplicates=n_dup),
                  exact_duplicates_removed_by_class=dup_removed_by_class,
                  non_numeric_cells_coerced_to_nan=n_coerced, infinite_cells_set_to_nan=n_inf,
                  missing_cells_remaining=int(df[fcols].isna().sum().sum()),
                  conflicting_feature_duplicate_groups_kept=n_conflict_groups,
                  columns_removed=removed, policy=dict(cc),
                  fraction_rows_removed=1 - len(df) / before["rows"])
    return df, report, pd.DataFrame(removed, columns=["column", "reason", "stage"]).assign(dataset=name)
