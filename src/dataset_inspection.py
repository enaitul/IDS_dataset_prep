"""Statistical inspection of the RAW datasets (before cleaning). All numbers come from the data."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from .label_mapping import normalize_label
from .utils import conflicting_duplicate_groups

ID_NAME_PATTERN = re.compile(r"(^|[^a-z])(id|ip|timestamp|time|date|port)([^a-z]|$)", re.I)

def _feature_cols(df: pd.DataFrame, label_col: str) -> list[str]:
    return [c for c in df.columns if not c.startswith("__") and c != label_col]

def _numeric_view(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].apply(pd.to_numeric, errors="coerce")

def identifier_candidates(df: pd.DataFrame, cols: list[str], label_col: str) -> list[dict]:
    out = []
    n = len(df)
    for c in cols:
        reasons = []
        if ID_NAME_PATTERN.search(c): reasons.append("name suggests identifier/time/port")
        s = df[c]
        if not pd.api.types.is_float_dtype(s) and n > 0 and s.nunique(dropna=True) / n > 0.9:
            reasons.append("near-unique values (>90% distinct)")
        if reasons:
            out.append(dict(column=c, reasons="; ".join(reasons)))
    return out

def inspect_dataset(name: str, df: pd.DataFrame, label_col: str, file_infos: list[dict], categorical_cols: list[str]) -> dict:
    feats = _feature_cols(df, label_col)
    num = _numeric_view(df, [c for c in feats if c not in categorical_cols])
    has_label = df[label_col].notna() & (df[label_col].astype(str).str.strip() != "")
    labels = df[label_col].where(has_label).map(lambda x: normalize_label(x, name) if isinstance(x, str) or pd.notna(x) else x)
    counts = labels.value_counts()                    # labelled rows only; unlabeled rows reported separately
    rows_without_label = int((~has_label).sum())
    # --- missing / inf
    miss = df[feats].isna().sum()
    inf_counts = {c: int(np.isinf(num[c].to_numpy(dtype=float, na_value=np.nan)).sum()) for c in num.columns}
    inf_counts = {c: v for c, v in inf_counts.items() if v}
    # numeric columns that failed numeric coercion (e.g. stray strings)
    coerced = {c: int(num[c].isna().sum() - df[c].isna().sum()) for c in num.columns}
    coerced = {c: v for c, v in coerced.items() if v > 0}
    # --- duplicates
    key_cols = feats + [label_col]
    dup_full = int(df.duplicated(subset=key_cols).sum())
    dup_feat = int(df.duplicated(subset=feats, keep=False).sum())
    n_conflict_groups = conflicting_duplicate_groups(df[has_label], feats, labels[has_label]) if len(feats) else 0
    # --- constant features
    const = [c for c in feats if df[c].nunique(dropna=True) <= 1]
    # --- imbalance
    maj, mnr = int(counts.max()), int(counts.min())
    imb = pd.DataFrame({"class": counts.index, "count": counts.values})
    imb["percent"] = 100 * imb["count"] / len(df)
    imb["ratio_to_majority"] = maj / imb["count"]
    schema = pd.DataFrame({
        "feature": feats,
        "dtype": [str(df[c].dtype) for c in feats],
        "role_guess": ["categorical" if c in categorical_cols else "numeric" for c in feats],
        "n_unique": [int(df[c].nunique(dropna=True)) for c in feats],
        "n_missing": [int(miss[c]) for c in feats],
        "n_inf": [inf_counts.get(c, 0) for c in feats],
        "is_constant": [c in const for c in feats],
    })
    summary = dict(
        dataset=name, files="; ".join(i["file"] for i in file_infos), formats=",".join(sorted({i["format"] for i in file_infos})),
        n_files=len(file_infos), n_records=int(len(df)), n_features_original=len(feats), label_column=label_col,
        n_classes=int(counts.size), rows_without_label=rows_without_label, missing_cells=int(miss.sum()), rows_with_any_missing=int(df[feats].isna().any(axis=1).sum()),
        inf_cells=int(sum(inf_counts.values())), columns_with_inf=len(inf_counts),
        exact_duplicate_rows=dup_full, duplicate_rows_pct=100 * dup_full / len(df),
        feature_duplicate_conflicting_groups=n_conflict_groups,
        n_constant_features=len(const), majority_class=str(counts.idxmax()), minority_class=str(counts.idxmin()),
        majority_count=maj, minority_count=mnr, imbalance_ratio_max_over_min=maj / mnr,
    )
    return dict(summary=summary, class_distribution=imb, schema=schema,
                missing_per_column=miss[miss > 0].to_dict(), inf_per_column=inf_counts, coerced_non_numeric=coerced,
                constant_features=const, identifier_candidates=identifier_candidates(df, feats, label_col),
                file_infos=file_infos)

def write_inspection_outputs(results: dict[str, dict], stats_dir) -> None:
    from pathlib import Path
    from .utils import write_json
    stats_dir = Path(stats_dir)
    if not results:
        return                                  # nothing inspected: write nothing rather than empty/misleading files
    stats_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r["summary"] for r in results.values()]).to_csv(stats_dir / "dataset_summary.csv", index=False)
    pd.concat([r["class_distribution"].assign(dataset=n) for n, r in results.items()]).to_csv(stats_dir / "class_distribution.csv", index=False)
    pd.concat([r["schema"].assign(dataset=n) for n, r in results.items()]).to_csv(stats_dir / "feature_schema_report.csv", index=False)
    pd.DataFrame([dict(dataset=n, column=c, n_missing=v, n_inf=r["inf_per_column"].get(c, 0))
                  for n, r in results.items() for c, v in {**{k: 0 for k in r["inf_per_column"]}, **r["missing_per_column"]}.items()]
                 ).to_csv(stats_dir / "missing_value_report.csv", index=False)
    pd.DataFrame([dict(dataset=n, n_records=r["summary"]["n_records"], exact_duplicate_rows=r["summary"]["exact_duplicate_rows"],
                       duplicate_rows_pct=r["summary"]["duplicate_rows_pct"],
                       conflicting_feature_duplicate_groups=r["summary"]["feature_duplicate_conflicting_groups"])
                  for n, r in results.items()]).to_csv(stats_dir / "duplicate_report.csv", index=False)
    pd.DataFrame([dict(dataset=n, **c) for n, r in results.items() for c in r["identifier_candidates"]]
                 ).to_csv(stats_dir / "identifier_leakage_candidates.csv", index=False)
    for n, r in results.items():
        write_json({k: v for k, v in r.items() if k not in ("class_distribution", "schema")}, stats_dir / f"{n}_inspection.json")
