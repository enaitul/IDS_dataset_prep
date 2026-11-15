"""Dataset loading + file validation. Never fabricates data: a missing dataset raises DatasetNotFound."""
from __future__ import annotations
import re, zipfile
from pathlib import Path
import pandas as pd
from .utils import md5sum, get_logger

log = get_logger()

NSL_COLUMNS = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
    "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_srv_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"]

DOWNLOAD_HELP = {
    "NSL-KDD": "KDDTrain+.txt and KDDTest+.txt, e.g. Zenodo record 10.5281/zenodo.17424143 (third-party mirror) -> data/raw/NSL-KDD/",
    "CICIDS2017": "GeneratedLabelledFlows.zip (8 CSVs) from https://www.unb.ca/cic/datasets/ids-2017.html -> data/raw/CICIDS2017/ (zip or extracted CSVs)",
    "InSDN": "InSDN_DatasetCSV.zip (Normal_data.csv, OVS.csv, metasploitable-2.csv) from https://aseados.ucd.ie/datasets/SDN/ -> data/raw/InSDN/ (zip or extracted CSVs)",
}

class DatasetNotFound(FileNotFoundError):
    pass

def ensure_extracted(raw_dir: Path) -> None:
    """Extract any .zip found in raw_dir (once) so users can drop the downloaded archives in directly."""
    for z in sorted(raw_dir.glob("*.zip")):
        target = raw_dir / f"_extracted_{z.stem}"
        if not target.exists():
            log.info("Extracting %s", z.name)
            with zipfile.ZipFile(z) as zf:
                zf.extractall(target)

def _files(raw_dir: Path, suffixes=(".csv", ".txt")) -> list[Path]:
    return sorted(f for f in raw_dir.rglob("*")
                  if f.is_file() and f.suffix.lower() in suffixes and "__MACOSX" not in f.parts and not f.name.startswith("."))

def _read_csv(path: Path, **kw) -> pd.DataFrame:
    last = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, **kw)
        except UnicodeDecodeError as e:
            last = e
    raise last

def _file_info(path: Path, rows: int, with_md5: bool = True) -> dict:
    return dict(file=path.name, path=str(path), format=path.suffix.lower().lstrip("."),
                size_bytes=path.stat().st_size, rows=int(rows), md5=md5sum(path) if with_md5 else None)

def load_nsl_kdd(raw_dir: Path, cfg_ds: dict) -> tuple[pd.DataFrame, list[dict]]:
    files = {f.name.lower(): f for f in _files(raw_dir, (".txt", ".csv"))}
    wanted = ["kddtrain+.txt", "kddtest+.txt"]
    missing = [w for w in wanted if w not in files]
    if missing:
        raise DatasetNotFound(f"NSL-KDD files missing in {raw_dir}: {missing}. Expected: {DOWNLOAD_HELP['NSL-KDD']}")
    frames, infos = [], []
    for w, src in zip(wanted, ["official_train", "official_test"]):
        with open(files[w], "r", encoding="utf-8", errors="replace") as f:
            first_line = f.readline()
        sep = "\t" if "\t" in first_line else ","
        df = pd.read_csv(files[w], sep=sep, header=None)
        if df.shape[1] not in (42, 43):
            raise ValueError(f"{files[w].name}: expected 42/43 columns (41 features + label [+ difficulty]), got {df.shape[1]}")
        df.columns = NSL_COLUMNS[: df.shape[1]]
        df["__source_file"] = files[w].name
        df["__official_split"] = src
        frames.append(df)
        info = _file_info(files[w], len(df))
        exp = (cfg_ds.get("expected_files") or {}).get(files[w].name.replace("kddtrain", "KDDTrain").replace("kddtest", "KDDTest"))
        if exp is None:
            exp = (cfg_ds.get("expected_files") or {}).get("KDDTrain+.txt" if "train" in w else "KDDTest+.txt")
        info["validation"] = {"rows_match_published": (exp is None) or info["rows"] == exp["rows"],
                              "md5_match_reference": (exp is None) or info["md5"] == exp["md5"],
                              "published_rows": exp and exp["rows"], "reference_md5": exp and exp["md5"]}
        infos.append(info)
    out = pd.concat(frames, ignore_index=True)
    out["__row_id"] = range(len(out))
    return out, infos

def _load_csv_folder(raw_dir: Path, name: str, expected_count: int | None) -> tuple[pd.DataFrame, list[dict]]:
    ensure_extracted(raw_dir)
    files = _files(raw_dir, (".csv",))
    if not files:
        raise DatasetNotFound(f"No CSV files for {name} in {raw_dir}. Expected: {DOWNLOAD_HELP[name]}")
    frames, infos = [], []
    for f in files:
        df = _read_csv(f)
        df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
        df["__source_file"] = f.name
        frames.append(df)
        info = _file_info(f, len(df))
        infos.append(info)
    # schema consistency check across files
    base = [c for c in frames[0].columns]
    for f, df in zip(files, frames):
        if set(df.columns) != set(base):
            log.warning("%s: column set differs from %s (%d vs %d columns) - concatenation will NaN-fill the difference",
                        f.name, files[0].name, df.shape[1], len(base))
    out = pd.concat(frames, ignore_index=True, sort=False)
    out["__row_id"] = range(len(out))
    if expected_count and len(files) != expected_count:
        log.warning("%s: found %d CSV files, expected %d", name, len(files), expected_count)
    for i in infos:
        i["validation"] = {"file_count_matches_expected": (expected_count is None) or len(files) == expected_count}
    return out, infos

def load_dataset(name: str, cfg: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Returns (raw dataframe incl. __row_id/__source_file, per-file info list)."""
    cfg_ds = cfg["datasets"][name]
    raw_dir = Path(cfg["_root"]) / cfg_ds["raw_dir"]
    if not raw_dir.exists():
        raise DatasetNotFound(f"{raw_dir} does not exist. Expected: {DOWNLOAD_HELP[name]}")
    if name == "NSL-KDD":
        return load_nsl_kdd(raw_dir, cfg_ds)
    df, infos = _load_csv_folder(raw_dir, name, cfg_ds.get("expected_file_count"))
    lab = cfg_ds["label_column"]
    if lab not in df.columns:
        cand = [c for c in df.columns if c.lower() == lab.lower()]
        if not cand:
            raise KeyError(f"{name}: label column '{lab}' not found. Columns: {list(df.columns)[:10]}...")
        df = df.rename(columns={cand[0]: lab})
    return df, infos

def available_datasets(cfg: dict) -> dict[str, bool]:
    """True when the dataset folder contains data files or an archive (does NOT validate them)."""
    res = {}
    for name, d in cfg["datasets"].items():
        raw = Path(cfg["_root"]) / d["raw_dir"]
        res[name] = raw.exists() and (any(raw.rglob("*.csv")) or any(raw.rglob("*.txt")) or any(raw.glob("*.zip")))
    return res
