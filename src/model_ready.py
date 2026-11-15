"""Reusable, leakage-safe model pipelines for Members 2 and 3.

Everything learned from data - imputation, encoding, (optionally) feature selection, SMOTE-NC, scaling - lives INSIDE
the pipeline, so cross-validation / out-of-fold stacking refits all of it on each training fold only. The held-out fold
is only transformed, never resampled.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold
from .preprocessing import TreePreprocessor, ColumnSelector, MLPEncoder
from .smote import SafeSMOTENC

class ContiguousLabelClassifier(BaseEstimator, ClassifierMixin):
    """Wrap estimators that need labels 0..m-1 (XGBoost) so folds missing a rare class do not crash."""
    def __init__(self, estimator=None):
        self.estimator = estimator
    def fit(self, X, y, **kw):
        self.classes_ = np.unique(y)
        lut = {c: i for i, c in enumerate(self.classes_)}
        self.est_ = clone(self.estimator).fit(X, np.array([lut[v] for v in y]), **kw)
        return self
    def predict_proba(self, X):
        return self.est_.predict_proba(X)
    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

def build_pipeline(estimator, spec: dict, selected_features: list[str] | None = None, representation: str = "tree",
                   smote: dict | None = None, category_labels: dict | None = None, seed: int = 42) -> Pipeline:
    """spec = {'numeric_cols': [...], 'categorical_cols': [...]} (feature_metadata.json). smote = smote_nc config dict or None."""
    steps = [("prep", TreePreprocessor(spec["numeric_cols"], spec["categorical_cols"]))]
    cat = list(spec["categorical_cols"])
    if selected_features is not None:
        steps.append(("select", ColumnSelector(selected_features)))
        cat = [c for c in cat if c in selected_features]
    if smote:
        steps.append(("smote", SafeSMOTENC(categorical_columns=cat, k_neighbors=smote["k_neighbors"],
                     target_per_class=smote["target_per_class"], max_ratio=smote["max_ratio"], min_samples=smote["min_samples"],
                     scale_for_neighbours=smote["scale_for_neighbours"], random_state=seed)))
    if representation == "mlp":
        steps.append(("encode", MLPEncoder(cat, category_labels)))
    steps.append(("clf", estimator))
    return Pipeline(steps)

def oof_probabilities(models: dict, X: pd.DataFrame, y, classes, n_splits: int = 5, seed: int = 42):
    """Out-of-fold class probabilities for stacking. `models` maps name -> UNFITTED pipeline from build_pipeline().
    Each fold clones and fits the whole pipeline on the fold's training part only (SMOTE-NC included); the held-out part
    is never oversampled. Classes with < n_splits members are pooled into one stratum for fold assignment only.
    Returns {name: array (n_rows, n_classes)} with columns ordered like `classes`."""
    y = np.asarray(y); classes = np.asarray(classes)
    vc = pd.Series(y).value_counts()
    strat = np.where(pd.Series(y).map(vc).to_numpy() < n_splits, "__rare__", y.astype(str))
    if (strat == "__rare__").sum() < n_splits:
        strat = np.where(strat == "__rare__", pd.Series(y).value_counts().idxmax().__str__(), strat)
    out = {n: np.zeros((len(y), len(classes))) for n in models}
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strat):
        for name, pipe in models.items():
            m = clone(pipe).fit(X.iloc[tr], y[tr])
            proba = m.predict_proba(X.iloc[va])
            cols = [int(np.flatnonzero(classes == c)[0]) for c in m.classes_]
            out[name][np.ix_(va, cols)] = proba
    return out
