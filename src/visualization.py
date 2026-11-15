"""Publication figures (PNG 300 dpi + PDF + SVG). All values are read from files produced by the pipeline - never typed in."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import p, read_json, get_logger

log = get_logger()
PAL = {"all_features": "#7f7f7f", "info_gain": "#1f77b4", "rfe": "#d95f02", "info_gain_rfe": "#1b9e77"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "figure.dpi": 100, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "font.family": "DejaVu Sans"})

def _save(fig, gdir: Path, stem: str):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(gdir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    log.info("figure: %s", stem)

def fig_class_distribution(cfg, gdir):
    f = p(cfg, "stats") / "class_distribution.csv"
    if not f.exists(): return
    t = pd.read_csv(f, keep_default_na=False, na_values=[""])
    for i, ds in enumerate(["CICIDS2017", "InSDN", "NSL-KDD"], 1):
        d = t[t.dataset == ds].sort_values("count")
        if d.empty: continue
        fig, ax = plt.subplots(figsize=(6.5, max(2.5, 0.32 * len(d) + 1)))
        ax.barh(d["class"], d["count"], color="#3b6ea8")
        ax.set_xscale("log"); ax.set_xlabel("Records (log scale, raw data before cleaning)"); ax.set_title(f"{ds}: class distribution")
        for y, (c, pc) in enumerate(zip(d["count"], d["percent"])):
            ax.text(c * 1.08, y, f"{c:,} ({pc:.2f}%)", va="center", fontsize=7)
        ax.set_xlim(right=d["count"].max() * 12)
        _save(fig, gdir, f"fig0{i}_class_distribution_{ds}")

def fig_missing(cfg, gdir):
    labels, counts, rates = [], [], []
    for ds in cfg["datasets"]:
        f = Path(cfg["_root"]) / cfg["paths"]["processed"] / "cleaned" / f"{ds}.parquet"
        if not f.exists():
            continue
        frame = pd.read_parquet(f)
        cols = [c for c in frame.columns if not c.startswith("__") and not c.startswith("label_")]
        if ds == "NSL-KDD" and "__official_split" in frame:
            groups = [("NSL-KDD KDDTrain+", frame[frame["__official_split"] == "official_train"]),
                      ("NSL-KDD KDDTest+", frame[frame["__official_split"] == "official_test"])]
        else:
            groups = [(ds, frame)]
        for label, part in groups:
            missing = int(part[cols].isna().sum().sum())
            cells = len(part) * len(cols)
            labels.append(label); counts.append(missing); rates.append(100 * missing / max(1, cells))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bars = ax.bar(labels, rates, color="#3b6ea8")
    ax.set_ylabel("Missing cells (% of features kept after cleaning)")
    ax.set_title("Missing feature cells after cleaning")
    ax.tick_params(axis="x", rotation=20)
    for bar, count, rate in zip(bars, counts, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{count:,} ({rate:.3g}%)", ha="center", va="bottom", fontsize=7)
    _save(fig, gdir, "fig04_missing_value_comparison")

def fig_cleaning_class_effect(cfg, gdir):
    for ds in cfg["datasets"]:
        f = p(cfg, "stats") / f"{ds}_cleaning_class_effect.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, index_col=0).fillna(0)
        d = d.sort_values("class_before")
        fig, ax = plt.subplots(figsize=(7, max(3, 0.28 * len(d) + 1)))
        y = np.arange(len(d)); h = 0.38
        ax.barh(y + h/2, d["class_before"], h, label="Before cleaning", color="#9e9e9e")
        ax.barh(y - h/2, d["class_after"], h, label="After cleaning", color="#3b6ea8")
        ax.set_yticks(y); ax.set_yticklabels(d.index.astype(str)); ax.set_xscale("log")
        ax.set_xlabel("Records (log scale)"); ax.set_title(f"{ds}: class counts before and after cleaning")
        ax.legend(frameon=False)
        for yi, removed, before in zip(y, d["removed"], d["class_before"]):
            pct = 100 * removed / before if before else 0
            ax.text(max(before, 1) * 1.08, yi + h/2, f"removed {removed:,.0f} ({pct:.2f}%)", va="center", fontsize=6)
        _save(fig, gdir, f"fig09_cleaning_class_effect_{ds}")

def fig_smote(cfg, gdir):
    reps = {ds: p(cfg, "stats") / f"{ds}_smote_nc_report.json" for ds in cfg["datasets"]}
    reps = {k: read_json(v) for k, v in reps.items() if v.exists()}
    for ds, r in reps.items():
        rep = r.get("all_features")
        if not rep: continue
        names = sorted(rep["before"], key=lambda k: rep["before"][k])
        b = [rep["before"][n] for n in names]; a = [rep["after"].get(n, rep["before"][n]) for n in names]
        fig, ax = plt.subplots(figsize=(6.5, max(2.8, 0.42 * len(names) + 1))); y = np.arange(len(names)); h = 0.4
        ax.barh(y + h / 2, b, h, label="Before SMOTE-NC (train)", color="#9e9e9e"); ax.barh(y - h / 2, a, h, label="After SMOTE-NC (train)", color="#1b9e77")
        ax.set_yticks(y); ax.set_yticklabels(names); ax.set_xscale("log"); ax.set_xlabel("Training records (log scale)")
        ax.set_title(f"{ds}: class counts before/after " + ("SMOTE-NC" if rep["method"].startswith("SMOTENC") else "SMOTE")); ax.legend(frameon=False, loc="lower right")
        skipped = rep.get("skipped_classes_too_small", {})
        if skipped:
            ax.text(0.01, -0.22 - 0.0, "Not oversampled (too few real samples): " + ", ".join(skipped), transform=ax.transAxes, fontsize=7, va="top")
        _save(fig, gdir, f"fig05_smote_before_after_{ds}")

def fig_rankings(cfg, gdir, top=20):
    for ds in cfg["datasets"]:
        f = p(cfg, "selected") / f"{ds}_info_gain_ranking.csv"
        if f.exists():
            d = pd.read_csv(f).head(top).iloc[::-1]
            fig, ax = plt.subplots(figsize=(6, 0.28 * len(d) + 1.2))
            ax.barh(d["feature"], d["info_gain"], color=np.where(d["selected"], PAL["info_gain"], "#bdbdbd"))
            ax.set_xlabel("Information gain (nats, mutual information)"); ax.set_title(f"{ds}: top-{len(d)} features by information gain (blue = selected)")
            _save(fig, gdir, f"fig06_information_gain_ranking_{ds}")
        f = p(cfg, "selected") / f"{ds}_rfe_ranking.csv"; c = p(cfg, "selected") / f"{ds}_rfe_k_tuning_curve.csv"
        if f.exists():
            d = pd.read_csv(f).head(top).iloc[::-1]
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 0.28 * len(d) + 1.4), gridspec_kw={"width_ratios": [1.2, 1]})
            a1.barh(d["feature"], d["rfe_rank"].max() + 1 - d["rfe_rank"], color=np.where(d["selected"], PAL["rfe"], "#bdbdbd"))
            a1.set_xlabel("RFE survival score (higher = eliminated later)"); a1.set_title(f"{ds}: top-{len(d)} features by RFE (orange = selected)")
            if c.exists():
                cv = pd.read_csv(c); a2.plot(cv["k"], cv["val_macro_f1"], marker="o", color=PAL["rfe"])
                k = read_json(p(cfg, "selected") / f"{ds}_selected_features.json")["rfe"]["chosen_k"]
                a2.axvline(k, ls="--", color="k", lw=0.8); a2.set_xlabel("Number of top-ranked features k"); a2.set_ylabel("Validation macro-F1 (RF proxy)")
                a2.set_title(f"Feature-count tuning (chosen k = {k})")
            _save(fig, gdir, f"fig07_rfe_ranking_{ds}")

def fig_counts(cfg, gdir):
    rows = []
    for ds in cfg["datasets"]:
        f = p(cfg, "selected") / f"{ds}_selected_features.json"
        if f.exists():
            for k, v in read_json(f)["counts"].items(): rows.append(dict(dataset=ds, config=k, n=v))
    if not rows: return
    t = pd.DataFrame(rows); dss = list(t.dataset.unique()); cfgs = list(PAL)
    fig, ax = plt.subplots(figsize=(6.5, 3.4)); w = 0.2; x = np.arange(len(dss))
    for i, c in enumerate(cfgs):
        vals = [t[(t.dataset == d) & (t.config == c)]["n"].sum() for d in dss]
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=c.replace("_", " "), color=PAL[c])
        ax.bar_label(bars, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(dss); ax.set_ylabel("Number of features"); ax.set_title("Feature count by selection method"); ax.legend(frameon=False, ncol=4, fontsize=7)
    _save(fig, gdir, "fig08_feature_count_comparison")

def make_all(cfg):
    g = p(cfg, "graphs")
    fig_class_distribution(cfg, g); fig_missing(cfg, g); fig_smote(cfg, g); fig_rankings(cfg, g); fig_counts(cfg, g); fig_cleaning_class_effect(cfg, g)
