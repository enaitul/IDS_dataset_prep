"""SMOTE-NC wrapper that is safe to place INSIDE a training-fold pipeline.

Applied only in fit (imblearn.Pipeline never calls fit_resample at predict time), so validation / test / held-out
OOF folds are never oversampled. Design decisions (all configurable, all recorded in `report_`):
  * Never balance up to the majority: target = min(target_per_class, max_ratio * n_class, n_majority).
  * Classes with < min_samples real rows are NOT oversampled (SMOTE needs k+1 neighbours; fabricating a class
    from a handful of points is not credible). They are listed in `skipped_classes_`; use class weights instead.
  * k_neighbors is reduced automatically so it never exceeds (smallest oversampled class - 1).
  * Numeric columns are MinMax-scaled for the neighbour search only; original rows are returned bit-for-bit unchanged
    (they occupy the first n_original_ rows), synthetic rows are mapped back to original units.
  * With no categorical column left (e.g. after feature selection) plain SMOTE is used and this is recorded.
"""
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler

class SafeSMOTENC(BaseEstimator):
    def __init__(self, categorical_columns=(), k_neighbors=5, target_per_class=50000, max_ratio=20,
                 min_samples=6, scale_for_neighbours=True, random_state=42):
        self.categorical_columns = categorical_columns
        self.k_neighbors = k_neighbors
        self.target_per_class = target_per_class
        self.max_ratio = max_ratio
        self.min_samples = min_samples
        self.scale_for_neighbours = scale_for_neighbours
        self.random_state = random_state

    def _strategy(self, counts: Counter):
        maj = max(counts.values())
        strat, skipped = {}, {}
        for cls, n in counts.items():
            target = int(min(self.target_per_class, self.max_ratio * n, maj))
            if target <= n:
                continue
            if n < self.min_samples:
                skipped[cls] = n
            else:
                strat[cls] = target
        return strat, skipped

    def fit(self, X, y=None):          # imblearn Pipeline compatibility
        return self

    def fit_resample(self, X: pd.DataFrame, y):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SafeSMOTENC needs a DataFrame so categorical columns are located by NAME")
        y = pd.Series(np.asarray(y)).reset_index(drop=True)
        X = X.reset_index(drop=True)
        counts = Counter(y.tolist())
        strat, skipped = self._strategy(counts)
        cols = list(X.columns)
        cat_idx = [i for i, c in enumerate(cols) if c in set(self.categorical_columns)]
        self.n_original_ = len(X)
        self.skipped_classes_ = {str(k): int(v) for k, v in skipped.items()}
        self.report_ = dict(method="SMOTENC" if cat_idx else "SMOTE (no categorical column present)",
                            categorical_columns=[cols[i] for i in cat_idx], random_state=self.random_state,
                            before={str(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: str(kv[0]))},
                            skipped_classes_too_small=self.skipped_classes_, params=dict(
                                k_neighbors_requested=self.k_neighbors, target_per_class=self.target_per_class,
                                max_ratio=self.max_ratio, min_samples=self.min_samples,
                                scale_for_neighbours=self.scale_for_neighbours))
        if not strat:
            self.n_synthetic_ = 0
            self.report_["after"] = self.report_["before"]; self.report_["k_neighbors_used"] = None
            return X, y.to_numpy()
        k = int(max(1, min(self.k_neighbors, min(counts[c] for c in strat) - 1)))
        num_idx = [i for i in range(len(cols)) if i not in cat_idx]
        Xw = X.copy()
        scaler = None
        if self.scale_for_neighbours and num_idx:
            scaler = MinMaxScaler().fit(X.iloc[:, num_idx])
            Xw.iloc[:, num_idx] = scaler.transform(X.iloc[:, num_idx])
        if cat_idx:
            sm = SMOTENC(categorical_features=cat_idx, sampling_strategy=strat, k_neighbors=k, random_state=self.random_state)
        else:
            sm = SMOTE(sampling_strategy=strat, k_neighbors=k, random_state=self.random_state)
        Xr, yr = sm.fit_resample(Xw.to_numpy(dtype="float64"), y.to_numpy())
        Xr = pd.DataFrame(Xr, columns=cols)
        n0 = len(X)
        syn = Xr.iloc[n0:].copy()
        if scaler is not None:
            syn.iloc[:, num_idx] = scaler.inverse_transform(syn.iloc[:, num_idx])
        out = pd.concat([X.astype("float64"), syn], ignore_index=True)   # originals untouched, synthetic appended last
        self.n_synthetic_ = len(out) - n0
        self.report_["k_neighbors_used"] = k
        self.report_["after"] = {str(k_): int(v) for k_, v in sorted(Counter(yr.tolist()).items(), key=lambda kv: str(kv[0]))}
        self.report_["n_synthetic"] = int(self.n_synthetic_)
        return out, yr
