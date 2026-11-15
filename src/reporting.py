"""Generates outputs/reports/Member1_Dataset_and_Feature_Engineering_Report.md from pipeline outputs ONLY.
Anything not produced by an actual run is printed as PENDING - nothing is estimated or typed in by hand."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .utils import p, read_json, feature_columns, conflicting_duplicate_groups

PENDING = "**PENDING** (dataset not processed yet - run `python run_pipeline.py`)"

def _md(df: pd.DataFrame, floatfmt="{:,.4g}") -> str:
    if df is None or df.empty: return "_(none)_"
    cols = list(df.columns)
    def f(v):
        if isinstance(v, float): return floatfmt.format(v)
        if isinstance(v, (int,)) and not isinstance(v, bool): return f"{v:,}"
        return str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(lines)

def _j(path):
    return read_json(path) if Path(path).exists() else None

def _csv(path, **kw):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], **kw) if Path(path).exists() else None

def gather(cfg):
    st, sel, spl = p(cfg, "stats"), p(cfg, "selected"), p(cfg, "splits")
    G = {}
    for ds in cfg["datasets"]:
        G[ds] = dict(inspect=_j(st / f"{ds}_inspection.json"), clean=_j(st / f"{ds}_cleaning_report.json"),
                     removed=_csv(st / f"{ds}_removed_columns.csv"), split=_j(spl / f"{ds}_split_indices.json"),
                     split_table=_csv(st / f"{ds}_split_class_table.csv"), audit=_j(st / f"{ds}_leakage_audit.json"),
                     sel=_j(sel / f"{ds}_selected_features.json"), smote=_j(st / f"{ds}_smote_nc_report.json"),
                     ig=_csv(sel / f"{ds}_info_gain_ranking.csv"), rfe=_csv(sel / f"{ds}_rfe_ranking.csv"))
    return G

DESC = {
 "CICIDS2017": "Canadian Institute for Cybersecurity IDS 2017: five days of emulated enterprise traffic converted to bidirectional flows by CICFlowMeter, with labelled benign traffic and DoS/DDoS, brute-force, web, botnet, infiltration and port-scan attacks. Not captured in an SDN environment.",
 "InSDN": "InSDN (Elsayed et al.): traffic generated in an SDN testbed (OpenFlow/Open vSwitch with an SDN controller) with attacks launched against Metasploitable-2 and OVS targets. The only one of the three datasets collected in an SDN setting.",
 "NSL-KDD": "NSL-KDD (Tavallaee et al.): a de-duplicated revision of KDD'99 with 41 connection-level features and attack classes grouped as DoS, Probe, R2L and U2R. Dated (1998-era traffic) and not SDN-based.",
}

def write_report(cfg) -> Path:
    G = gather(cfg); out = p(cfg, "reports") / "Member1_Dataset_and_Feature_Engineering_Report.md"
    done = [d for d, g in G.items() if g["clean"] and g["split"]]
    L = []; A = L.append
    A("# Member 1 - Dataset Preparation & Feature Engineering Report\n")
    A("*Project: An Enhanced Intrusion Detection System for SDN Using Hybrid Neural Model and Ensemble Learning.*\n")
    A("> Every number in this report was read from files written by `run_pipeline.py`. Sections for datasets that have not been processed are marked **PENDING**.\n")
    A(f"Datasets processed end-to-end: **{', '.join(done) if done else 'none'}**. Not processed: **{', '.join(d for d in cfg['datasets'] if d not in done) or 'none'}**.\n")
    A("## 0. Specification conflicts (see also `Spec_Conflicts.md`)\n")
    A("The manuscript describes a CNN-RNN/LSTM hybrid with SMOTE and a train/test split, whereas the implementation brief specifies MLP + RF + XGBoost + Logistic-Regression stacking, SMOTE-NC and a 70:15:15 split. This pipeline implements the brief and exports 2-D tabular data that a CNN/LSTM can reshape later. The manuscript's Algorithm (Steps 2-5) normalises and applies SMOTE *before* splitting, which leaks information; this pipeline splits first.\n")
    A("## 1. Dataset descriptions\n")
    for d in cfg["datasets"]:
        A(f"**{d}.** {DESC[d]}\n")
    external_rows = None
    nsl_inspection = G.get("NSL-KDD", {}).get("inspect")
    nsl_clean = G.get("NSL-KDD", {}).get("clean")
    nsl_split = G.get("NSL-KDD", {}).get("split")
    if nsl_inspection and nsl_clean and nsl_split:
        train_rows = sum(nsl_split["metadata"]["sizes"].values())
        external_path = Path(cfg["_root"]) / cfg["paths"]["processed"] / "NSL-KDD" / "external_test_binary" / "tree" / "all_features" / "y_test.parquet"
        external_rows = len(pd.read_parquet(external_path)) if external_path.exists() else None
        A(f"**NSL-KDD split protocol.** The multiclass split uses KDDTrain+ only ({train_rows:,} rows). KDDTest+ is a separate external binary test set ({external_rows:,} rows after duplicate removal, read from the generated external-test labels).\n" if external_rows is not None else f"**NSL-KDD split protocol.** The multiclass split uses KDDTrain+ only ({train_rows:,} rows). KDDTest+ external binary test count: {PENDING}.\n")
    A("## 2. Dataset statistics (source row counts; NSL-KDD missing and duplicate counts are split by official source)\n")
    rows = []
    for d, g in G.items():
        if g["inspect"]:
            s = g["inspect"]["summary"]
            if d == "NSL-KDD" and nsl_clean:
                cleaned_path = Path(cfg["_root"]) / cfg["paths"]["processed"] / "cleaned" / "NSL-KDD.parquet"
                if cleaned_path.exists():
                    frame = pd.read_parquet(cleaned_path)
                    cols = feature_columns(frame)
                    info_by_file = {f["file"]: f["rows"] for f in g["inspect"]["file_infos"]}
                    split_names = {"official_train": "KDDTrain+", "official_test": "KDDTest+"}
                    for source, display in split_names.items():
                        part = frame[frame["__official_split"] == source]
                        feature_dups = part.duplicated(subset=cols, keep=False).sum()
                        row = {"Dataset / source": display, "Records": info_by_file.get(display + ".txt", len(part)),
                               "Features": nsl_clean["after"]["columns"],
                               "Classes": part["label_original"].nunique(), "Missing cells": int(part[cols].isna().sum().sum()),
                               "Inf cells": 0, "Exact duplicates removed": info_by_file.get(display + ".txt", len(part)) - len(part),
                               "Feature-duplicate rows": int(feature_dups),
                               "Conflicting duplicate groups": conflicting_duplicate_groups(part, cols, part["label_original"]) if source == "official_train" else "—"}
                        rows.append(row)
                    note = "KDDTest+ lacks the difficulty column; missing-cell counts cover only features retained after cleaning."
                    if frame.loc[frame["__official_split"] == "official_test", "__difficulty"].isna().all():
                        A(note + "\n")
            else:
                rows.append({"Dataset / source": d, "Records": s["n_records"], "Features": s["n_features_original"], "Classes": s["n_classes"],
                             "Missing cells": s["missing_cells"], "Inf cells": s["inf_cells"], "Exact duplicates removed": s["exact_duplicate_rows"],
                             "Feature-duplicate rows": "—", "Conflicting duplicate groups": s["feature_duplicate_conflicting_groups"]})
    A(_md(pd.DataFrame(rows)) if rows else PENDING); A("")
    A("### Source files and validation\n")
    for d, g in G.items():
        if g["inspect"]:
            A(f"**{d}**\n"); A(_md(pd.DataFrame([{k: v for k, v in f.items() if k in ("file", "format", "size_bytes", "rows", "md5")} for f in g["inspect"]["file_infos"]]))); 
            val = [f.get("validation", {}) for f in g["inspect"]["file_infos"]]
            bad = [v for v in val if v and not all(x for k, x in v.items() if k.endswith(("match_published", "match_reference", "matches_expected")) )]
            if d in ("CICIDS2017", "InSDN"):
                expected = cfg["datasets"][d].get("expected_file_count")
                status = "file count matches expected; no published row-count/checksum reference available" if expected == len(val) else "file count does not match expected; see inspection JSON"
                A(f"\nValidation: {status}.\n")
            else:
                A(f"\nValidation against published reference values: {'all checks passed' if val and not bad else ('MISMATCH - see inspection JSON' if bad else 'n/a')}.\n")
    A("## 3. Original attack categories and record counts\n")
    for d, g in G.items():
        A(f"### {d}\n")
        t = _csv(p(cfg, "stats") / "class_distribution.csv")
        A(_md(t[t.dataset == d][["class", "count", "percent", "ratio_to_majority"]]) if t is not None and (t.dataset == d).any() else PENDING); A("")
    A("## 4. Label harmonisation\n")
    m = _csv(p(cfg, "metadata") / "label_mapping.csv")
    if m is not None:
        A("Full table: `data/metadata/label_mapping.csv`. Labels that are NOT confidently harmonised (UNMAPPED, medium or unknown confidence): `outputs/statistics/labels_not_confidently_harmonized.csv`.\n")
        A(_md(m[m.observed_in_data][["dataset", "original_label", "harmonized_label", "confidence"]]))
        A("\nUnmapped labels are retained for per-dataset experiments and excluded only from cross-dataset (harmonised) experiments. No attack types were merged merely to align datasets.\n")
        A("For NSL-KDD, the label `attack` is a coarse binary label from KDDTest+; it is not a multiclass attack subtype and is therefore not harmonised.\n")
    else: A(PENDING)
    A("## 5. Data cleaning\n")
    for d, g in G.items():
        c = g["clean"]; A(f"### {d}\n")
        if not c: A(PENDING + "\n"); continue
        A(f"Rows {c['before']['rows']:,} -> {c['after']['rows']:,} ({100*c['fraction_rows_removed']:.2f}% removed). Removed rows: {json.dumps(c['rows_removed'])}. Infinite cells set to NaN: {c['infinite_cells_set_to_nan']:,}. Non-numeric cells coerced to NaN: {c['non_numeric_cells_coerced_to_nan']:,}. Missing cells remaining for train-fitted imputation: {c['missing_cells_remaining']:,}. Conflicting feature-duplicate groups (same features, different labels) kept and reported: {c['conflicting_feature_duplicate_groups_kept']:,}.\n")
        effect = _csv(p(cfg, "stats") / f"{d}_cleaning_class_effect.csv")
        if effect is not None:
            class_col = effect.columns[0]
            effect = effect.rename(columns={class_col: "class"})
            effect["percent_removed"] = 100 * effect["removed"] / effect["class_before"].replace(0, pd.NA)
            effect = effect.rename(columns={"class_before": "before", "class_after": "after", "percent_removed": "% removed"})
            A("Post-cleaning class effect (counts before and after; rows removed; percent removed):\n")
            A(_md(effect[["class", "before", "after", "removed", "% removed"]], floatfmt="{:,.2f}")); A("")
        A("Removed columns:\n"); A(_md(g["removed"][["column", "stage", "reason"]]) if g["removed"] is not None else ""); A("")
    A("Policy: exact duplicates = identical on all features and original label, first occurrence kept (file order). Duplicates are identified after removing identifier columns (Flow ID, IPs, ports, timestamp), so flows differing only in those fields count as duplicates. Missing values are never imputed during cleaning.\n")
    A("## 6. Splitting strategy\n")
    A("Per-class seeded random split on the original label (manual stratification) to 70/15/15; seed in `configs/config.yaml`. Row identifiers (`__row_id`) of every split are in `data/splits/<dataset>_split_indices.json`.\n")
    for d, g in G.items():
        A(f"### {d}\n")
        if g["split"]:
            m_ = g["split"]["metadata"]
            A(f"Sizes: {m_['sizes']}; achieved fractions: { {k: round(v,4) for k,v in m_['achieved_fractions'].items()} }.\n")
            A("Classes too small for exact 70/15/15 stratification:\n"); A(_md(pd.DataFrame(m_["classes_too_small_for_exact_stratification"])))
            if g["split_table"] is not None: A("\nPer-class counts per split:\n"); A(_md(g["split_table"].rename(columns={g["split_table"].columns[0]: "class"})))
        else: A(PENDING)
        A("")
    A("## 7. Leakage prevention and audit\n")
    A("* Split precedes every fitted step; imputers, vocabularies, scalers and selectors are fitted on the training split only.\n* SMOTE-NC is applied to the training split only (and inside every training fold for CV/stacking via `src/model_ready.py`).\n* Validation labels are used only to choose the RFE feature count.\n* Unseen categories map to code -1 (tree) / all-zero one-hot (MLP).\n")
    for d, g in G.items():
        if g["audit"]:
            a = g["audit"]; A(f"**{d}** - cross-split audit: near-duplicates (rounded to {a['near_duplicate_rounding_decimals']} decimals) of test rows found in train: {a['near_duplicate_rows_test_in_train']:,} ({a['near_duplicate_pct_of_test_in_train']:.2f}% of test); of validation rows found in train: {a['near_duplicate_rows_validation_in_train']:,} ({a['near_duplicate_pct_of_validation_in_train']:.2f}%). {a['temporal_note']}\n")
    A("## 8-9. Feature selection (Information Gain, RFE)\n")
    for d, g in G.items():
        A(f"### {d}\n")
        if g["sel"]:
            s = g["sel"]
            A(f"* Data: {s['selection_data']}.\n* Information Gain: {s['info_gain']['criterion']}; estimator `{s['info_gain']['estimator']}`, n_neighbors={s['info_gain']['n_neighbors']}, subsample rows={s['info_gain']['subsample_rows']:,}. Selected {len(s['info_gain']['features'])} of {s['n_features_total']}.\n* RFE: {s['rfe']['estimator']}, step={s['rfe']['step']}, subsample rows={s['rfe']['subsample_rows']:,}, k grid={s['rfe']['k_grid']}, tolerance={s['rfe']['tolerance']} on {s['rfe']['tuning_metric']}. Chosen k={s['rfe']['chosen_k']}.\n* IG->RFE hybrid: chosen k={s['info_gain_rfe']['chosen_k']} from {s['info_gain_rfe']['n_after_info_gain']} IG-retained features.\n")
            A("Feature counts: " + json.dumps(s["counts"]) + "\n")
        else: A(PENDING + "\n")
    A("## 10. SMOTE-NC\n")
    for d, g in G.items():
        A(f"### {d}\n")
        if g["smote"] and g["smote"].get("all_features"):
            r = g["smote"]["all_features"]
            A(f"Method used: {r['method']}; categorical columns: {r['categorical_columns']}; k_neighbors used: {r['k_neighbors_used']}; seed {r['random_state']}; params {r['params']}.\n")
            A(f"Not oversampled (fewer than min_samples real rows): {r['skipped_classes_too_small'] or 'none'}.\n")
            A(_md(pd.DataFrame({"class": list(r["before"]), "train_before": list(r["before"].values()), "train_after": [r["after"].get(k, v) for k, v in r["before"].items()]}))); A("")
        else: A(PENDING + "\n")
    A("## 11. Selected features\n")
    for d, g in G.items():
        A(f"### {d}\n")
        if g["sel"]:
            for k in ("info_gain", "rfe", "info_gain_rfe"):
                A(f"**{k}** ({len(g['sel'][k]['features'])}): {', '.join(g['sel'][k]['features'])}\n")
        else: A(PENDING + "\n")
    A("## 12. Experimental parameters\n")
    cfg_pub = {k: v for k, v in cfg.items() if k not in ("_root", "identifier_columns")}
    A("```json\n" + json.dumps(cfg_pub, indent=2) + "\n```\n")
    A("## 13. Figures and tables generated\n")
    gd = p(cfg, "graphs"); figs = sorted(x.name for x in gd.glob("*.png"))
    A("\n".join(f"* `outputs/graphs/{f}` (+ `.pdf`, `.svg`)" for f in figs) if figs else PENDING)
    A("\nTables in `outputs/statistics/`: dataset_summary.csv, class_distribution.csv, missing_value_report.csv, duplicate_report.csv, feature_schema_report.csv, identifier_leakage_candidates.csv, per-dataset cleaning/split/leakage/SMOTE reports. Rankings in `outputs/selected_features/`.\n")
    A("## 14. Limitations and unresolved issues\n")
    small_train = []
    benign_oversampled = []
    for d, g in G.items():
        if g["split_table"] is not None:
            table = g["split_table"]
            class_col = table.columns[0]
            small_train.extend(f"{d} {r[class_col]} ({int(r['train'])})" for _, r in table[table["train"] < 30].iterrows())
        rep = (g["smote"] or {}).get("all_features") if g["smote"] else None
        if rep:
            benign_oversampled.extend(f"{d} {label} ({rep['before'][label]:,} to {rep['after'].get(label, rep['before'][label]):,})"
                                      for label in rep["before"] if label.lower() in ("normal", "benign") and rep["after"].get(label, rep["before"][label]) > rep["before"][label])
    A("* Random stratified splitting of flow records can be optimistic (flows of one attack run are correlated). Temporal/group-aware evaluation is recommended as a robustness check; see the leakage audit above.\n* Only InSDN was collected in an SDN setting; CICIDS2017 and NSL-KDD generalise to SDN only indirectly.\n* Information gain uses a k-NN mutual-information estimator on a stratified subsample (see parameters); RFE uses a Random-Forest estimator on a subsample. Results may vary with other estimators.\n* SMOTE-NC interpolates numeric features, so synthetic count-valued features can be non-integers. Classes with too few real samples are not oversampled; use class weights for them.\n* `Destination Port` is kept (CICIDS2017/InSDN) but is a possible shortcut feature; an ablation without it is advised.\n* NSL-KDD multiclass experiments use KDDTrain+ only; KDDTest+ is kept separate for binary external evaluation.\n")
    A("* Classes with fewer than 30 real training rows have unreliable per-class metrics: " + (", ".join(small_train) if small_train else "none in the saved split tables") + ".\n")
    A("* SMOTE-NC also oversamples benign classes when they are below the fixed target: " + (", ".join(benign_oversampled) if benign_oversampled else "none in the saved SMOTE reports") + ".\n")
    ddos_effect = _csv(p(cfg, "stats") / "InSDN_cleaning_class_effect.csv")
    if ddos_effect is not None:
        ccol = ddos_effect.columns[0]
        ddos = ddos_effect[ddos_effect[ccol].astype(str).str.lower().eq("ddos")]
        if not ddos.empty:
            r = ddos.iloc[0]
            A(f"* InSDN DDoS loses most flows to deduplication: {int(r['class_before']):,} before, {int(r['class_after']):,} after, {int(r['removed']):,} removed.\n")
    A("* NSL-KDD files, if taken from a third-party mirror, should be cited as such.\n")
    A("---\n# Research-paper-ready text\n")
    def n(d, *path, default="[PENDING]"):
        g = G[d]; x = g
        try:
            for k in path: x = x[k]
            return x
        except Exception: return default
    A("### Dataset\n")
    parts = []
    for d in cfg["datasets"]:
        if not G[d]["clean"]:
            parts.append(f"{d} [PENDING]")
        elif d == "NSL-KDD" and nsl_split and external_rows is not None:
            parts.append(f"{d} (multiclass KDDTrain+ split: {train_rows:,} rows; {n(d,'clean','after','columns')} features; external binary KDDTest+: {external_rows:,} rows after duplicate removal)")
        else:
            parts.append(f"{d} ({n(d,'clean','after','rows'):,} records after cleaning, {n(d,'clean','after','columns')} features)")
    A("The proposed system is trained and evaluated on three public benchmark datasets: " + "; ".join(parts) + ". Each dataset is processed independently and keeps its own label set; a harmonised label view is provided only for explicitly defined cross-dataset experiments.\n")
    if nsl_inspection and nsl_split and external_rows is not None:
        A("The NSL-KDD multiclass split uses KDDTrain+ only; KDDTest+ is held out as a separate external binary test set. The `attack` label is a coarse binary label from KDDTest+.\n")
    A("### Data Preprocessing\n")
    A("Duplicate records (identical features and label) were removed, infinite values were converted to missing values, and identifier, timestamp and constant columns were removed with each removal documented. The data were then divided into 70% training, 15% validation and 15% test subsets using a seeded per-class stratified split. Missing values were imputed with training-set medians (numeric) or modes (categorical); categorical attributes were ordinal-encoded for tree-based models and one-hot encoded for the MLP; and numeric features were Min-Max scaled (MLP only), with every statistic fitted on the training subset only and re-used unchanged on validation and test data.\n")
    A("### Feature Selection\n")
    c1 = "; ".join(f"{d}: IG {n(d,'sel','counts','info_gain')}, RFE {n(d,'sel','counts','rfe')}, IG-RFE {n(d,'sel','counts','info_gain_rfe')} of {n(d,'sel','counts','all_features')} features" for d in cfg["datasets"])
    A(f"Features were ranked by information gain (mutual information with the class label) and by Recursive Feature Elimination with a Random-Forest estimator, both computed on training data only. The number of features retained by RFE was selected by validation macro-F1. Resulting feature counts - {c1}.\n")
    A("### Class Balancing\n")
    A("Class imbalance was addressed with SMOTE-NC applied exclusively to the training data - and, for cross-validation and out-of-fold stacking, only to the training portion of each fold - so that no synthetic sample reaches validation or test data. Minority classes were oversampled toward a capped target (never above the majority-class size and at most a fixed multiple of their real size); classes with too few real samples for a reliable neighbourhood were left unchanged. Exact parameters are listed in Section 12.\n")
    out.write_text("\n".join(L), encoding="utf-8")
    return out
