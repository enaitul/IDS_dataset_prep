"""Shared helpers: config, paths, hashing, JSON IO."""
from __future__ import annotations
import hashlib, json, logging
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = str(path.resolve().parents[1])
    return cfg

def p(cfg: dict, key: str) -> Path:
    """Resolve a configured path relative to the project root; create it."""
    out = Path(cfg["_root"]) / cfg["paths"][key]
    out.mkdir(parents=True, exist_ok=True)
    return out

def md5sum(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        return super().default(o)

def write_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_Enc, ensure_ascii=False)

def read_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)

def get_logger(name: str = "sdn_ids") -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
        log.addHandler(h)
        log.setLevel(logging.INFO)
    return log

def feature_columns(df) -> list[str]:
    """Model features = every column that is not metadata (__*) or a label (label_*)."""
    return [c for c in df.columns if not c.startswith("__") and not c.startswith("label_")]

def row_hash(df, cols):
    """64-bit hash per row over `cols` (NaN-aware). Scales to millions of rows, unlike groupby on dozens of columns."""
    import pandas as pd
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy()

def conflicting_duplicate_groups(df, feature_cols, label_series) -> int:
    """# of distinct feature vectors that appear with more than one label (label noise). Hash-based."""
    import pandas as pd
    g = pd.DataFrame({"h": row_hash(df, feature_cols), "y": pd.Series(label_series).to_numpy()}).dropna()
    return int((g.groupby("h")["y"].nunique() > 1).sum())
