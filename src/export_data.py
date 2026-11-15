"""Model-ready exports + the loader Members 2/3 use (no knowledge of the internals required)."""
from __future__ import annotations
from pathlib import Path
import joblib, json
import numpy as np
import pandas as pd

SPLITS = ("train", "validation", "test")

def _w(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def export_feature_config(out_dir: Path, rep: str, config: str, X: dict, y: dict, Xsm=None, ysm=None, synth_mask=None):
    """X/y: dicts keyed train/validation/test. Optional SMOTE-NC-resampled TRAIN only (never val/test)."""
    base = out_dir / rep / config
    for s in SPLITS:
        _w(X[s], base / f"X_{s}.parquet")
        _w(pd.DataFrame({"y": y[s]}), base / f"y_{s}.parquet")
    if Xsm is not None:
        _w(Xsm, base / "X_train_smotenc.parquet")
        _w(pd.DataFrame({"y": ysm, "is_synthetic": synth_mask}), base / "y_train_smotenc.parquet")

def export_labels(out_dir: Path, labels: dict[str, pd.DataFrame]):
    for s, df in labels.items():
        _w(df, out_dir / "labels" / f"labels_{s}.parquet")

def load_export(root: str | Path, dataset: str, representation: str = "tree", features: str = "all_features", smote: bool = False) -> dict:
    """Load one exported configuration.

    >>> d = load_export("data/processed", "NSL-KDD", representation="tree", features="rfe")
    >>> d["X_train"], d["y_train"], d["X_validation"], d["y_validation"], d["X_test"], d["y_test"]
    With smote=True, X_train / y_train are the SMOTE-NC-resampled training set (validation/test are untouched).
    """
    root = Path(root) / dataset
    base = root / representation / features
    if not base.exists():
        raise FileNotFoundError(f"{base} not found. Available: {[p.name for p in (root / representation).glob('*')] if (root / representation).exists() else 'nothing exported yet'}")
    d = {}
    for s in ("validation", "test"):
        d[f"X_{s}"] = pd.read_parquet(base / f"X_{s}.parquet")
        d[f"y_{s}"] = pd.read_parquet(base / f"y_{s}.parquet")["y"].to_numpy()
    if smote:
        d["X_train"] = pd.read_parquet(base / "X_train_smotenc.parquet")
        ym = pd.read_parquet(base / "y_train_smotenc.parquet")
        d["y_train"] = ym["y"].to_numpy(); d["is_synthetic"] = ym["is_synthetic"].to_numpy()
    else:
        d["X_train"] = pd.read_parquet(base / "X_train.parquet")
        d["y_train"] = pd.read_parquet(base / "y_train.parquet")["y"].to_numpy()
    art = root / "artifacts"
    d["class_mapping"] = json.loads((art / "class_mapping.json").read_text())
    d["feature_metadata"] = json.loads((art / "feature_metadata.json").read_text())
    d["feature_names"] = list(d["X_train"].columns)
    d["labels"] = {s: pd.read_parquet(root / "labels" / f"labels_{s}.parquet") for s in SPLITS}
    d["label_encoder"] = joblib.load(art / "label_encoder.joblib")
    d["preprocessor"] = joblib.load(art / "tree_preprocessor.joblib")
    return d

def load_cleaned_split(cfg_or_root, dataset: str) -> dict:
    """Cleaned but NOT imputed/encoded frames for each split (+ encoded multiclass y), for strictly fold-safe stacking:
    feed X_train_raw to model_ready.build_pipeline() so imputation, encoding, selection and SMOTE-NC are ALL refitted per fold.
    `cfg_or_root` = the dict from src.utils.load_config()."""
    import joblib
    from .utils import feature_columns, read_json
    cfg = cfg_or_root
    root = Path(cfg["_root"])
    df = pd.read_parquet(root / cfg["paths"]["processed"] / "cleaned" / f"{dataset}.parquet").set_index("__row_id", drop=False)
    sp = read_json(root / cfg["paths"]["splits"] / f"{dataset}_split_indices.json")
    le = joblib.load(root / cfg["paths"]["processed"] / dataset / "artifacts" / "label_encoder.joblib")
    cols = feature_columns(df); out = {}
    for s in SPLITS:
        part = df.loc[sp[s]]
        out[f"X_{s}_raw"] = part[cols].reset_index(drop=True)
        out[f"y_{s}"] = le.transform(part["label_original"])
    return out

def load_external_test(root: str | Path, dataset: str = "NSL-KDD", representation: str = "tree", features: str = "all_features") -> dict:
    """Load the external binary-only test set (e.g. KDDTest+ for NSL-KDD)."""
    base = Path(root) / dataset / "external_test_binary" / representation / features
    if not base.exists():
        raise FileNotFoundError(f"{base} not found. Check that external_test_binary was exported.")
    X = pd.read_parquet(base / "X_test.parquet")
    y = pd.read_parquet(base / "y_test.parquet")["y"].to_numpy()
    return {"X_test": X, "y_test": y, "y_binary": y}
