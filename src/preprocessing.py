"""Leakage-controlled preprocessing. Every statistic (medians, modes, category vocabularies, min/max) is learned in
fit() from the data passed to fit() - which must be the TRAINING split only - and re-used unchanged in transform().

Two model-ready representations:
  * 'tree' : TreePreprocessor        -> median-imputed numerics (UNSCALED) + ordinal-coded categoricals.
                                        Right for RF / XGBoost, and the common base that SMOTE-NC operates on.
  * 'mlp'  : TreePreprocessor + MLPEncoder -> MinMax-scaled numerics (clipped to [0,1]) + one-hot categoricals.
Unseen categories map to code -1 (tree) / an all-zero one-hot block (mlp); they never raise.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

UNSEEN_CODE = -1.0

class TreePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols=(), categorical_cols=()):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        num = [c for c in self.numeric_cols if c in X.columns]
        cat = [c for c in self.categorical_cols if c in X.columns]
        numX = X[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        self.dropped_constant_ = [c for c in num if numX[c].nunique(dropna=True) <= 1] + \
                                 [c for c in cat if X[c].nunique(dropna=True) <= 1]
        self.numeric_cols_ = [c for c in num if c not in self.dropped_constant_]
        self.categorical_cols_ = [c for c in cat if c not in self.dropped_constant_]
        self.medians_ = numX[self.numeric_cols_].median().to_dict()
        self.categories_, self.modes_ = {}, {}
        for c in self.categorical_cols_:
            s = X[c].dropna().astype(str)
            self.categories_[c] = sorted(s.unique().tolist())
            self.modes_[c] = s.mode().iloc[0]
        self.feature_names_out_ = self.numeric_cols_ + self.categorical_cols_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)
        out = X[self.numeric_cols_].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        out = out.fillna(self.medians_).astype("float64")
        for c in self.categorical_cols_:
            s = X[c].astype("object")
            isna = s.isna()
            mapping = {cat: float(i) for i, cat in enumerate(self.categories_[c])}
            s = s.where(~isna, self.modes_[c]).astype(str)
            out[c] = s.map(mapping).fillna(UNSEEN_CODE).astype("float64")
        return out[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select (and order) columns by NAME so selected sets apply identically to every split."""
    def __init__(self, columns=()):
        self.columns = columns
    def fit(self, X, y=None):
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise KeyError(f"Selected features absent from input: {missing[:5]}")
        return self
    def transform(self, X):
        return X[list(self.columns)]
    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.columns))

class MLPEncoder(BaseEstimator, TransformerMixin):
    """MinMax(numeric, clipped) + one-hot(categorical) on top of the tree representation."""
    def __init__(self, categorical_cols=(), category_labels=None):
        self.categorical_cols = categorical_cols
        self.category_labels = category_labels

    def fit(self, X: pd.DataFrame, y=None):
        self.cat_ = [c for c in self.categorical_cols if c in X.columns]
        self.num_ = [c for c in X.columns if c not in self.cat_]
        self.scaler_ = MinMaxScaler(clip=True).fit(X[self.num_]) if self.num_ else None
        labels = self.category_labels or {}
        self.onehot_ = []                                   # (source column, code, output name)
        for c in self.cat_:
            for code in sorted({v for v in X[c].unique() if v >= 0}):
                names = labels.get(c)
                self.onehot_.append((c, code, f"{c}__{names[int(code)] if names else int(code)}"))
        self.feature_names_out_ = list(self.num_) + [n for _, _, n in self.onehot_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if self.num_:
            parts.append(pd.DataFrame(self.scaler_.transform(X[self.num_]), columns=self.num_, index=X.index))
        if self.onehot_:
            parts.append(pd.DataFrame({n: (X[c].to_numpy() == code).astype("float64") for c, code, n in self.onehot_}, index=X.index))
        return pd.concat(parts, axis=1)[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)

def feature_spec(df: pd.DataFrame, categorical_candidates: list[str]) -> dict:
    cols = [c for c in df.columns if not c.startswith("__") and not c.startswith("label_")]
    cat = [c for c in cols if c in categorical_candidates]
    return dict(numeric_cols=[c for c in cols if c not in cat], categorical_cols=cat)
