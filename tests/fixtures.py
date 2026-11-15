"""TEST-ONLY synthetic fixtures that mimic the *schema quirks* of the three datasets.
They exist solely to exercise the code. They are NEVER used for any reported result, figure or statistic."""
from __future__ import annotations
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.data_loader import NSL_COLUMNS

ROOT = Path(__file__).resolve().parents[1]

def _signal(rng, n, shift, k):
    return rng.normal(shift, 1.0, size=(n, k))

def make_nsl(raw_dir: Path, seed=0):
    rng = np.random.default_rng(seed)
    classes = {"normal": 900, "neptune": 500, "smurf": 300, "satan": 120, "guess_passwd": 40, "buffer_overflow": 5, "rootkit": 2}
    rows = []
    for i, (lab, n) in enumerate(classes.items()):
        X = np.abs(_signal(rng, n, i, 38)).round(3)
        df = pd.DataFrame(X, columns=[c for c in NSL_COLUMNS[:41] if c not in ("protocol_type", "service", "flag")])
        df["protocol_type"] = rng.choice(["tcp", "udp", "icmp"], n, p=[.6, .3, .1] if i % 2 else [.3, .3, .4])
        df["service"] = rng.choice(["http", "ftp", "smtp", "private", "ecr_i"], n)
        df["flag"] = rng.choice(["SF", "S0", "REJ"], n)
        df["label"] = lab; df["difficulty"] = rng.integers(1, 22, n)
        rows.append(df)
    df = pd.concat(rows, ignore_index=True)[NSL_COLUMNS]
    df["wrong_fragment"] = 5.0                 # constant column
    df = pd.concat([df, df.iloc[:25]], ignore_index=True)   # 25 exact duplicates
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    cut = int(len(df) * 0.85)
    df.iloc[:cut].to_csv(raw_dir / "KDDTrain+.txt", header=False, index=False)
    df.iloc[cut:].to_csv(raw_dir / "KDDTest+.txt", header=False, index=False)

def _flow_frame(rng, labels: dict, with_ports=True):
    parts = []
    for i, (lab, n) in enumerate(labels.items()):
        X = np.abs(_signal(rng, n, i * 0.8, 10)).round(4)
        df = pd.DataFrame(X, columns=[f" Feature {j}" for j in range(10)])
        df[" Protocol"] = rng.choice([6, 17, 1], n)
        df[" Destination Port"] = rng.choice([80, 443, 22, 21], n)
        df[" Label"] = lab
        parts.append(df)
    return pd.concat(parts, ignore_index=True)

def make_cicids(raw_dir: Path, seed=1):
    rng = np.random.default_rng(seed)
    labels = {"BENIGN": 1200, "DDoS": 500, "PortScan": 300, "Bot": 60, "Web Attack \u2013 XSS": 30, "Infiltration": 4, "Heartbleed": 2}
    df = _flow_frame(rng, labels).sample(frac=1, random_state=seed).reset_index(drop=True)
    df["Flow ID"] = [f"10.0.0.{i%250}-192.168.1.{i%200}-{i}" for i in range(len(df))]
    df["Source IP"] = "10.0.0.1"; df["Destination IP"] = "192.168.1.2"
    df["Source Port"] = rng.integers(1024, 65535, len(df))
    df["Timestamp"] = pd.date_range("2017-07-03 09:00", periods=len(df), freq="s").strftime("%d/%m/%Y %H:%M:%S")
    df[" Fwd Header Length"] = rng.integers(0, 500, len(df)); df[" Fwd Header Length.1"] = df[" Fwd Header Length"]
    df[" Const Col"] = 0
    df.loc[3, " Feature 1"] = np.inf; df.loc[7, " Feature 2"] = -np.inf; df.loc[11, " Feature 3"] = np.nan
    df.loc[20:22, :] = np.nan                  # entirely empty rows
    df = pd.concat([df, df.iloc[30:45]], ignore_index=True)   # 15 exact duplicates
    raw_dir.mkdir(parents=True, exist_ok=True)
    half = len(df) // 2
    df.iloc[:half].to_csv(raw_dir / "Monday-WorkingHours.pcap_ISCX.csv", index=False, encoding="cp1252")
    df.iloc[half:].to_csv(raw_dir / "Tuesday-WorkingHours.pcap_ISCX.csv", index=False, encoding="cp1252")

def make_insdn(raw_dir: Path, seed=2):
    rng = np.random.default_rng(seed)
    labels = {"Normal": 800, "DDoS": 400, "DoS": 300, "Probe": 200, "BFA": 60, "Web-Attack": 25, "BOTNET": 20, "U2R": 4}
    df = _flow_frame(rng, labels).sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.rename(columns={" Label": "Label", " Protocol": "Protocol", " Destination Port": "Dst Port"})
    df["Flow ID"] = [f"f{i}" for i in range(len(df))]; df["Src IP"] = "10.0.0.5"; df["Dst IP"] = "10.0.0.9"
    df["Src Port"] = rng.integers(1024, 65535, len(df))
    df["Timestamp"] = pd.date_range("2020-01-01", periods=len(df), freq="s").strftime("%d/%m/%Y %I:%M:%S %p")
    raw_dir.mkdir(parents=True, exist_ok=True)
    third = len(df) // 3
    df.iloc[:third].to_csv(raw_dir / "Normal_data.csv", index=False)
    df.iloc[third:2 * third].to_csv(raw_dir / "OVS.csv", index=False)
    df.iloc[2 * third:].to_csv(raw_dir / "metasploitable-2.csv", index=False)

def make_workspace(tmp: Path, datasets=("NSL-KDD", "CICIDS2017", "InSDN")) -> dict:
    """Copy config into tmp/configs and create synthetic raw data under tmp/data/raw. Returns loaded cfg."""
    (tmp / "configs").mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())
    cfg["feature_selection"]["rfe"].update(k_grid=[3, 5, 8], n_estimators=20, proxy_n_estimators=20)
    cfg["smote_nc"]["target_per_class"] = 800
    (tmp / "configs" / "config.yaml").write_text(yaml.safe_dump(cfg))
    from src.utils import load_config
    c = load_config(tmp / "configs" / "config.yaml")
    mk = {"NSL-KDD": make_nsl, "CICIDS2017": make_cicids, "InSDN": make_insdn}
    for d in datasets:
        mk[d](tmp / cfg["datasets"][d]["raw_dir"])
    return c
