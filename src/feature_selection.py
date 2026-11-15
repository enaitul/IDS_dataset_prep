"""Information Gain and RFE feature selection. Every function here must be called with TRAINING data only; the only
use of validation data is choosing the feature COUNT for RFE (curve saved to disk). IG uses no validation labels."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.metrics import f1_score

def stratified_subsample(X: pd.DataFrame, y: np.ndarray, max_rows: int, seed: int, min_per_class: int = 500):
    """Seeded subsample keeping every small class (up to min_per_class rows) and the remaining classes proportional."""
    y = np.asarray(y)
    if len(y) <= max_rows:
        return X, y
    rng = np.random.default_rng(seed)
    frac = max_rows / len(y)
    keep = []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        n = max(min(len(idx), min_per_class), int(round(frac * len(idx))))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return X.iloc[keep], y[keep]

def info_gain_ranking(X: pd.DataFrame, y, categorical_cols: list[str], n_neighbors: int, seed: int) -> pd.DataFrame:
    """Information gain = mutual information I(feature; class). Categorical (ordinal-coded) columns use the discrete
    estimator, numeric columns the k-NN continuous estimator (Kraskov); seeded because sklearn adds jitter."""
    mask = np.array([c in set(categorical_cols) for c in X.columns])
    mi = mutual_info_classif(X.to_numpy(dtype="float64"), np.asarray(y), discrete_features=mask,
                             n_neighbors=n_neighbors, random_state=seed)
    df = pd.DataFrame({"feature": X.columns, "info_gain": mi}).sort_values("info_gain", ascending=False, kind="mergesort").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    tot = df["info_gain"].sum()
    df["share"] = df["info_gain"] / tot if tot > 0 else 0.0
    df["cumulative_share"] = df["share"].cumsum()
    return df

def select_by_cumulative_mass(ranking: pd.DataFrame, mass: float) -> list[str]:
    pos = ranking[ranking["info_gain"] > 0]
    if pos.empty:
        return []
    k = int(np.searchsorted(pos["cumulative_share"].to_numpy(), mass - 1e-12) + 1)
    return pos["feature"].iloc[:k].tolist()

def _rf(n_estimators, max_depth, seed):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1, random_state=seed,
                                  class_weight="balanced_subsample")

def rfe_ranking(X: pd.DataFrame, y, n_estimators: int, max_depth: int, step: int, seed: int) -> pd.DataFrame:
    """Full RFE elimination order (n_features_to_select=1). Rank 1 = last survivor. With step>1 features removed in the
    same round share a rank. Estimator: RandomForest(balanced_subsample)."""
    rfe = RFE(_rf(n_estimators, max_depth, seed), n_features_to_select=1, step=step).fit(X.to_numpy(dtype="float64"), np.asarray(y))
    df = pd.DataFrame({"feature": X.columns, "rfe_rank": rfe.ranking_})
    return df.sort_values(["rfe_rank", "feature"], kind="mergesort").reset_index(drop=True)

def tune_feature_count(ranked: list[str], Xtr, ytr, Xva, yva, grid: list[int], tolerance: float, n_estimators: int, seed: int):
    """Validation macro-F1 for the top-k features; pick the SMALLEST k within `tolerance` of the best score."""
    rows = []
    for k in sorted({min(g, len(ranked)) for g in grid} | {len(ranked)}):
        cols = ranked[:k]
        m = _rf(n_estimators, None, seed).fit(Xtr[cols], ytr)
        pred = m.predict(Xva[cols])
        rows.append(dict(k=k, val_macro_f1=f1_score(yva, pred, average="macro", labels=np.unique(yva), zero_division=0)))
    curve = pd.DataFrame(rows)
    best = curve["val_macro_f1"].max()
    chosen = int(curve.loc[curve["val_macro_f1"] >= best - tolerance, "k"].min())
    return curve, chosen

class InfoGainSelector(BaseEstimator, TransformerMixin):
    """Sklearn-compatible IG selector so selection can be re-fitted INSIDE every CV fold (strictest leakage control)."""
    def __init__(self, categorical_cols=(), cumulative_mass=0.95, n_neighbors=3, max_rows=100000, random_state=42):
        self.categorical_cols = categorical_cols; self.cumulative_mass = cumulative_mass
        self.n_neighbors = n_neighbors; self.max_rows = max_rows; self.random_state = random_state
    def fit(self, X, y):
        Xs, ys = stratified_subsample(X, np.asarray(y), self.max_rows, self.random_state)
        self.ranking_ = info_gain_ranking(Xs, ys, list(self.categorical_cols), self.n_neighbors, self.random_state)
        self.selected_ = select_by_cumulative_mass(self.ranking_, self.cumulative_mass) or list(X.columns[:1])
        return self
    def transform(self, X):
        return X[self.selected_]
    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_)

class RFESelector(BaseEstimator, TransformerMixin):
    """RFE (RandomForest) selector with a fixed feature count k (e.g. the validation-tuned one)."""
    def __init__(self, k=20, n_estimators=60, max_depth=12, step=1, max_rows=30000, random_state=42):
        self.k = k; self.n_estimators = n_estimators; self.max_depth = max_depth
        self.step = step; self.max_rows = max_rows; self.random_state = random_state
    def fit(self, X, y):
        Xs, ys = stratified_subsample(X, np.asarray(y), self.max_rows, self.random_state)
        self.ranking_ = rfe_ranking(Xs, ys, self.n_estimators, self.max_depth, self.step, self.random_state)
        self.selected_ = self.ranking_["feature"].iloc[: self.k].tolist()
        return self
    def transform(self, X):
        return X[self.selected_]
    def get_feature_names_out(self, input_features=None):
        return np.array(self.selected_)
