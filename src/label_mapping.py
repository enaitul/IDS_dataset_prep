"""Label harmonisation.

Two label views are kept for every record:
  * label_original   - dataset-specific label (text-normalised only)
  * label_harmonized - common category, or UNMAPPED when no justified equivalence exists
plus label_binary (0 = benign, 1 = any attack, incl. UNMAPPED attacks).

Nothing is merged just to make datasets look compatible. UNMAPPED rows stay in the data for
per-dataset experiments and are excluded only from cross-dataset (harmonised) experiments.
"""
from __future__ import annotations
import re
import pandas as pd

UNMAPPED = "UNMAPPED"
UNKNOWN_REASON = "Label not present in the documented mapping table; needs manual review"

def normalize_label(x, dataset: str | None = None) -> str:
    """Text-normalise a raw label: fix the CICIDS2017 mojibake dash, collapse whitespace. For NSL-KDD, lowercase."""
    s = str(x).strip()
    s = re.sub(r"[^\x20-\x7e]+", "-", s)          # '\ufffd' / '\x96' in 'Web Attack <?> Brute Force'
    s = re.sub(r"\s*-\s*", " - ", s) if "Web Attack" in s else s
    s = re.sub(r"\s+", " ", s).strip()
    if dataset == "NSL-KDD":
        s = s.lower()
    return s

def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

# (original label, harmonised, confidence, reason)
_NSL_DOS = ["back", "land", "neptune", "pod", "smurf", "teardrop", "apache2", "udpstorm", "processtable", "mailbomb"]
_NSL_PROBE = ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]
_NSL_R2L = ["ftp_write", "imap", "phf", "multihop", "warezmaster", "warezclient", "spy", "xlock", "xsnoop",
            "snmpguess", "snmpgetattack", "httptunnel", "sendmail", "named", "worm"]
_NSL_U2R = ["buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack", "xterm", "ps"]

_RAW = {"NSL-KDD": [("normal", "Benign", "high", "Normal traffic")]}
_RAW["NSL-KDD"] += [(l, "DoS", "high", "KDD taxonomy DoS: resource-exhaustion from a single logical source (no DDoS class exists in NSL-KDD)") for l in _NSL_DOS]
_RAW["NSL-KDD"] += [(l, "Probe", "high", "KDD taxonomy Probe: scanning / surveillance") for l in _NSL_PROBE]
_RAW["NSL-KDD"] += [("guess_passwd", "BruteForce", "medium", "Online password guessing against a service; comparable to CICIDS Patator / InSDN BFA but via different tooling (R2L category in KDD)")]
_RAW["NSL-KDD"] += [(l, UNMAPPED, "none", "R2L (remote-to-local) host-exploitation / misc. attack; no equivalent category in CICIDS2017 or InSDN") for l in _NSL_R2L]
_RAW["NSL-KDD"] += [(l, UNMAPPED, "none", "U2R (user-to-root) privilege escalation; host-level, not equivalent to any CICIDS2017 category (InSDN U2R is a different attack set)") for l in _NSL_U2R]

_RAW["CICIDS2017"] = [
    ("BENIGN", "Benign", "high", "Normal traffic"),
    ("DoS Hulk", "DoS", "high", "Application-layer HTTP flood from one source"),
    ("DoS GoldenEye", "DoS", "high", "Application-layer HTTP DoS from one source"),
    ("DoS slowloris", "DoS", "high", "Slow-connection DoS from one source"),
    ("DoS Slowhttptest", "DoS", "high", "Slow-rate HTTP DoS from one source"),
    ("DDoS", "DDoS", "high", "Distributed flood (LOIC) from multiple sources"),
    ("Heartbleed", UNMAPPED, "none", "OpenSSL memory-disclosure exploit; not a flood and not equivalent to any common category (only 11 flows in the public CSVs per literature; verify)"),
    ("PortScan", "Probe", "high", "Port scanning"),
    ("FTP-Patator", "BruteForce", "high", "FTP credential brute force"),
    ("SSH-Patator", "BruteForce", "high", "SSH credential brute force"),
    ("Bot", "Botnet", "high", "Ares botnet C&C traffic"),
    ("Web Attack - Brute Force", "WebAttack", "medium", "HTTP login brute force; could alternatively be BruteForce - kept with other web attacks to respect the dataset's own grouping"),
    ("Web Attack - XSS", "WebAttack", "high", "Cross-site scripting"),
    ("Web Attack - Sql Injection", "WebAttack", "high", "SQL injection"),
    ("Infiltration", "Infiltration", "high", "Inside-network infiltration (Metasploit / Cool disk)"),
]
_RAW["InSDN"] = [
    ("Normal", "Benign", "high", "Normal traffic"),
    ("DoS", "DoS", "high", "Single-source DoS"),
    ("DDoS", "DDoS", "high", "Distributed DoS"),
    ("Probe", "Probe", "high", "Reconnaissance / scanning"),
    ("BFA", "BruteForce", "high", "Brute-force attack on services (FTP/SSH/HTTP login)"),
    ("Web-Attack", "WebAttack", "high", "Web application attacks (SQLi/XSS/...)"),
    ("BOTNET", "Botnet", "high", "Botnet traffic"),
    ("U2R", UNMAPPED, "none", "Privilege escalation via exploit tooling; NSL-KDD also has U2R but with different attack instances - equivalence not established, so not merged"),
]

HARMONIZED_CLASSES = ["Benign", "DoS", "DDoS", "Probe", "BruteForce", "Botnet", "WebAttack", "Infiltration"]

_LOOKUP = {ds: {_key(o): (o, h, c, r) for (o, h, c, r) in rows} for ds, rows in _RAW.items()}

def lookup(dataset: str, label: str):
    return _LOOKUP[dataset].get(_key(label))

def harmonize(dataset: str, label: str) -> str:
    hit = lookup(dataset, label)
    return hit[1] if hit else UNMAPPED

def is_benign(dataset: str, label: str) -> bool:
    hit = lookup(dataset, label)
    return bool(hit and hit[1] == "Benign")

def build_mapping_table(observed: dict[str, list[str]]) -> pd.DataFrame:
    """Mapping CSV: documented mappings + any observed-but-unknown labels flagged for review."""
    rows = []
    for ds, labels in observed.items():
        seen = set()
        for lab in labels:
            hit = lookup(ds, lab)
            if hit:
                rows.append(dict(dataset=ds, original_label=lab, harmonized_label=hit[1], confidence=hit[2], reason=hit[3], observed_in_data=True))
            else:
                rows.append(dict(dataset=ds, original_label=lab, harmonized_label=UNMAPPED, confidence="unknown", reason=UNKNOWN_REASON, observed_in_data=True))
            seen.add(_key(lab))
        for (o, h, c, r) in _RAW.get(ds, []):          # documented but not observed
            if _key(o) not in seen:
                rows.append(dict(dataset=ds, original_label=o, harmonized_label=h, confidence=c, reason=r, observed_in_data=False))
    return pd.DataFrame(rows)

def add_label_columns(df: pd.DataFrame, dataset: str, raw_label_col: str) -> pd.DataFrame:
    out = df.copy()
    out["label_original"] = out[raw_label_col].map(lambda x: normalize_label(x, dataset))
    out = out.drop(columns=[raw_label_col])
    out["label_harmonized"] = out["label_original"].map(lambda s: harmonize(dataset, s))
    out["label_binary"] = (~out["label_original"].map(lambda s: is_benign(dataset, s))).astype("int8")
    return out
