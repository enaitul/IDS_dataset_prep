"""How Members 2 and 3 consume Member 1's outputs. No knowledge of the preprocessing internals is required.

    python examples/consume_exports.py NSL-KDD rfe          # dataset, feature set

Feature sets: all_features | info_gain | rfe | info_gain_rfe.  Representations: 'tree' (RF/XGBoost) or 'mlp' (MLP).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from src.export_data import load_export, load_cleaned_split
from src.model_ready import build_pipeline, oof_probabilities, ContiguousLabelClassifier
from src.utils import load_config

def main(dataset="NSL-KDD", features="rfe", root=None):
    cfg = load_config(root) if root else load_config()
    proc = Path(cfg["_root"]) / cfg["paths"]["processed"]
    # 1) Ready-to-train arrays. smote=True swaps in the SMOTE-NC-resampled TRAIN set; validation/test are never resampled.
    tree = load_export(proc, dataset, "tree", features, smote=True)
    mlp = load_export(proc, dataset, "mlp", features, smote=True)
    print("classes:", tree["class_mapping"]["classes"])
    print("train", tree["X_train"].shape, "synthetic rows:", int(tree["is_synthetic"].sum()), "| val", tree["X_validation"].shape, "| test", tree["X_test"].shape)

    # 2) Base learners on the exported arrays (tune on VALIDATION; touch TEST only once at the very end).
    rf = RandomForestClassifier(200, n_jobs=-1, random_state=cfg["seed"]).fit(tree["X_train"], tree["y_train"])
    nn = MLPClassifier((64, 32), max_iter=200, random_state=cfg["seed"]).fit(mlp["X_train"], mlp["y_train"])
    for name, m, d in (("RF", rf, tree), ("MLP", nn, mlp)):
        print(f"{name} validation macro-F1: {f1_score(d['y_validation'], m.predict(d['X_validation']), average='macro'):.4f}")

    # 3) Out-of-fold stacking with SMOTE-NC INSIDE each training fold (Member 2/3: do NOT stack on the pre-resampled arrays).
    plain = load_export(proc, dataset, "tree", features, smote=False)                # REAL training rows only
    meta = plain["feature_metadata"]; cols = meta["feature_sets"][features]
    raw = load_cleaned_split(cfg, dataset)                                              # cleaned, NOT imputed: every fold refits imputer/encoder/SMOTE-NC
    raw_like, y_raw = raw["X_train_raw"], raw["y_train"]
    spec = {"numeric_cols": [c for c in meta["numeric_cols"]], "categorical_cols": list(meta["categorical_cols"])}
    models = {"rf": build_pipeline(RandomForestClassifier(100, n_jobs=-1, random_state=0), spec, cols, "tree", cfg["smote_nc"], seed=cfg["seed"]),
              "xgb": build_pipeline(ContiguousLabelClassifier(XGBClassifier(n_estimators=100, max_depth=6, verbosity=0)), spec, cols, "tree", cfg["smote_nc"], seed=cfg["seed"]),
              "mlp": build_pipeline(MLPClassifier((64, 32), max_iter=100, random_state=0), spec, cols, "mlp", cfg["smote_nc"], seed=cfg["seed"])}
    classes = np.arange(tree["class_mapping"]["n_classes"])
    oof = oof_probabilities(models, raw_like, y_raw, classes, n_splits=5, seed=cfg["seed"])
    meta_X = np.hstack([oof[k] for k in models])
    meta_clf = LogisticRegression(max_iter=1000).fit(meta_X, y_raw)
    print("stacked meta-classifier fitted on OOF probabilities:", meta_X.shape)

if __name__ == "__main__":
    main(*(sys.argv[1:3]))
