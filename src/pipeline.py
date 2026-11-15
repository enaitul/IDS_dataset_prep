"""Orchestration of Member 1's pipeline. Each step reads the previous step's artifacts from disk, so steps can be re-run."""
from __future__ import annotations
import platform, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn, imblearn

from .utils import p, write_json, read_json, get_logger, feature_columns
from .data_loader import load_dataset, available_datasets
from .dataset_inspection import inspect_dataset, write_inspection_outputs
from .cleaning import clean_dataset
from .label_mapping import build_mapping_table
from .splitting import stratified_split_indices, split_class_table, leakage_audit
from .preprocessing import TreePreprocessor, MLPEncoder, feature_spec
from .feature_selection import (stratified_subsample, info_gain_ranking, select_by_cumulative_mass, rfe_ranking,
                                tune_feature_count)
from .smote import SafeSMOTENC
from .export_data import export_feature_config, export_labels, SPLITS
from sklearn.preprocessing import LabelEncoder

log = get_logger()

# ---------------------------------------------------------------- inspection
def step_inspect(cfg, datasets):
    results = {}
    meta = p(cfg, "metadata")
    for ds in datasets:
        raw, infos = load_dataset(ds, cfg)
        lab = cfg["datasets"][ds]["label_column"]
        log.info("[%s] inspecting %d raw records", ds, len(raw))
        results[ds] = inspect_dataset(ds, raw, lab, infos, cfg["datasets"][ds]["categorical_columns"])
        joblib.dump(results[ds], meta / f"{ds}_inspection.joblib")
        del raw
    allres = {d: joblib.load(meta / f"{d}_inspection.joblib") for d in cfg["datasets"] if (meta / f"{d}_inspection.joblib").exists()}
    write_inspection_outputs(allres, p(cfg, "stats"))
    return allres

# ---------------------------------------------------------------- cleaning
def cleaned_path(cfg, ds): return p(cfg, "processed") / "cleaned" / f"{ds}.parquet"

def step_clean(cfg, datasets):
    stats = p(cfg, "stats")
    for ds in datasets:
        raw, _ = load_dataset(ds, cfg)
        df, report, removed = clean_dataset(ds, raw, cfg, copy=False)
        cleaned_path(cfg, ds).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cleaned_path(cfg, ds), index=False)
        write_json(report, stats / f"{ds}_cleaning_report.json")
        removed.to_csv(stats / f"{ds}_removed_columns.csv", index=False)
        b, a = report["before"]["class_counts"], report["after"]["class_counts"]
        pd.DataFrame({"class_before": pd.Series(b), "class_after": pd.Series(a)}).fillna(0).astype(int).assign(
            removed=lambda t: t.class_before - t.class_after).to_csv(stats / f"{ds}_cleaning_class_effect.csv")
        log.info("[%s] cleaned: %d -> %d rows, %d columns removed", ds, report["before"]["rows"], report["after"]["rows"], len(removed))
        del raw, df
    _write_mapping(cfg)

def _write_mapping(cfg):
    obs = {}
    for ds in cfg["datasets"]:
        f = p(cfg, "stats") / f"{ds}_cleaning_report.json"
        if f.exists():
            r = read_json(f)
            from .label_mapping import normalize_label
            obs[ds] = sorted({normalize_label(k, ds) for k in r["before"]["class_counts"]})
    if obs:
        t = build_mapping_table(obs)
        t.to_csv(p(cfg, "metadata") / "label_mapping.csv", index=False)
        t[t["harmonized_label"].eq("UNMAPPED") | t["confidence"].isin(["medium", "unknown"])].to_csv(
            p(cfg, "stats") / "labels_not_confidently_harmonized.csv", index=False)

# ---------------------------------------------------------------- splitting
def split_path(cfg, ds): return p(cfg, "splits") / f"{ds}_split_indices.json"

def step_split(cfg, datasets):
    sp = cfg["split"]
    for ds in datasets:
        df = pd.read_parquet(cleaned_path(cfg, ds))
        use_train_only = (
            cfg["datasets"][ds].get("multiclass_source") == "official_train_only"
            and "__official_split" in df.columns
        )
        if use_train_only:
            split_mask = (df["__official_split"] == "official_train").to_numpy()
            df_split = df.loc[split_mask].reset_index(drop=True)
        else:
            df_split = df
        idx, meta = stratified_split_indices(df_split["label_original"].to_numpy(), cfg["seed"], (sp["train"], sp["validation"], sp["test"]))
        rid = df_split["__row_id"].to_numpy()
        out = {k: rid[v].tolist() for k, v in idx.items()}
        if "__official_split" in df.columns:
            all_rid = df["__row_id"].to_numpy()
            out["official_train_row_ids"] = all_rid[(df["__official_split"] == "official_train").to_numpy()].tolist()
            out["official_test_row_ids"] = all_rid[(df["__official_split"] == "official_test").to_numpy()].tolist()
        out["metadata"] = meta
        write_json(out, split_path(cfg, ds))
        split_class_table(df_split["label_original"], idx).to_csv(p(cfg, "stats") / f"{ds}_split_class_table.csv")
        audit = leakage_audit(df_split, idx, feature_columns(df), sp["near_duplicate_decimals"])
        write_json(audit, p(cfg, "stats") / f"{ds}_leakage_audit.json")
        log.info("[%s] split sizes %s", ds, meta["sizes"])

# ---------------------------------------------------------------- shared context
class Context:
    """Loads cleaned data + split, fits the TRAIN-only preprocessor and returns tree-representation frames."""
    def __init__(self, cfg, ds):
        self.cfg, self.ds = cfg, ds
        df = pd.read_parquet(cleaned_path(cfg, ds)).set_index("__row_id", drop=False)
        sp = read_json(split_path(cfg, ds))
        self.df, self.sp = df, sp
        self.rows = {s: df.loc[sp[s]] for s in SPLITS}
        self.spec = feature_spec(df, cfg["datasets"][ds]["categorical_columns"])
        tr = self.rows["train"]
        self.le = LabelEncoder().fit(tr["label_original"])
        split_df = pd.concat([self.rows[s] for s in SPLITS])
        unseen = set(split_df["label_original"]) - set(self.le.classes_)
        assert not unseen, f"classes absent from train: {unseen}"
        self.y = {s: self.le.transform(self.rows[s]["label_original"]) for s in SPLITS}
        self.prep = TreePreprocessor(self.spec["numeric_cols"], self.spec["categorical_cols"]).fit(tr[feature_columns(df)])
        self.X = {s: self.prep.transform(self.rows[s][feature_columns(df)]).reset_index(drop=True) for s in SPLITS}
        self.cat = list(self.prep.categorical_cols_)

# ---------------------------------------------------------------- feature selection
def step_select(cfg, datasets):
    fs = cfg["feature_selection"]; seed = cfg["seed"]; out = p(cfg, "selected")
    for ds in datasets:
        c = Context(cfg, ds)
        Xtr, ytr = c.X["train"], c.y["train"]
        # --- Information Gain (train only; no validation labels)
        Xs, ys = stratified_subsample(Xtr, ytr, fs["stratified_subsample_rows"]["info_gain"], seed, fs["min_rows_per_class_in_subsample"])
        ig = info_gain_ranking(Xs, ys, c.cat, fs["info_gain"]["n_neighbors"], seed)
        ig_sel = select_by_cumulative_mass(ig, fs["info_gain"]["cumulative_mass"])
        ig.assign(selected=ig["feature"].isin(ig_sel)).to_csv(out / f"{ds}_info_gain_ranking.csv", index=False)
        # --- RFE (train only) + validation-tuned k
        Xr, yr = stratified_subsample(Xtr, ytr, fs["stratified_subsample_rows"]["rfe"], seed, fs["min_rows_per_class_in_subsample"])
        Xv, yv = stratified_subsample(c.X["validation"], c.y["validation"], 200000, seed, fs["min_rows_per_class_in_subsample"])
        r = fs["rfe"]
        def run_rfe(cols, tag):
            rk = rfe_ranking(Xr[cols], yr, r["n_estimators"], r["max_depth"], r["step"], seed)
            curve, k = tune_feature_count(rk["feature"].tolist(), Xr, yr, Xv, yv, r["k_grid"], r["tolerance"], r["proxy_n_estimators"], seed)
            curve.to_csv(out / f"{ds}_{tag}_k_tuning_curve.csv", index=False)
            return rk, curve, k
        rk, curve, k = run_rfe(list(Xtr.columns), "rfe")
        rfe_sel = rk["feature"].tolist()[:k]
        rk.assign(selected=rk["feature"].isin(rfe_sel)).to_csv(out / f"{ds}_rfe_ranking.csv", index=False)
        # --- IG -> RFE hybrid (RFE restricted to IG-retained features)
        rk2, curve2, k2 = run_rfe(ig_sel, "info_gain_rfe") if len(ig_sel) > 1 else (rk, curve, k)
        hyb_sel = rk2["feature"].tolist()[:k2]
        rk2.assign(selected=rk2["feature"].isin(hyb_sel)).to_csv(out / f"{ds}_info_gain_rfe_ranking.csv", index=False)
        res = dict(dataset=ds, seed=seed, n_features_total=Xtr.shape[1],
                   selection_data="TRAIN split only (stratified subsample); validation used ONLY to choose RFE feature count",
                   info_gain=dict(criterion=f"smallest top-k with cumulative IG share >= {fs['info_gain']['cumulative_mass']} (IG>0)",
                                  estimator="mutual_info_classif (k-NN continuous / discrete for categorical)",
                                  n_neighbors=fs["info_gain"]["n_neighbors"], subsample_rows=int(len(ys)), features=ig_sel),
                   rfe=dict(estimator=f"RandomForest(n_estimators={r['n_estimators']}, max_depth={r['max_depth']}, class_weight=balanced_subsample)",
                            step=r["step"], subsample_rows=int(len(yr)), k_grid=r["k_grid"], tolerance=r["tolerance"], chosen_k=int(k),
                            tuning_metric="validation macro-F1 (RandomForest proxy)", features=rfe_sel),
                   info_gain_rfe=dict(chosen_k=int(k2), n_after_info_gain=len(ig_sel), features=hyb_sel),
                   counts={"all_features": Xtr.shape[1], "info_gain": len(ig_sel), "rfe": len(rfe_sel), "info_gain_rfe": len(hyb_sel)})
        write_json(res, out / f"{ds}_selected_features.json")
        log.info("[%s] feature counts %s", ds, res["counts"])

# ---------------------------------------------------------------- export
def step_export(cfg, datasets):
    ex = cfg["export"]; sm = cfg["smote_nc"]; seed = cfg["seed"]
    for ds in datasets:
        c = Context(cfg, ds)
        sel = read_json(p(cfg, "selected") / f"{ds}_selected_features.json")
        out = p(cfg, "processed") / ds
        art = out / "artifacts"; art.mkdir(parents=True, exist_ok=True)
        feats_cfg = {"all_features": list(c.X["train"].columns), "info_gain": sel["info_gain"]["features"],
                     "rfe": sel["rfe"]["features"], "info_gain_rfe": sel["info_gain_rfe"]["features"]}
        export_labels(out, {s: pd.DataFrame({"row_id": c.rows[s]["__row_id"].to_numpy(), "label_original": c.rows[s]["label_original"].to_numpy(),
                             "y_multiclass": c.y[s], "y_binary": c.rows[s]["label_binary"].to_numpy(),
                             "label_harmonized": c.rows[s]["label_harmonized"].to_numpy()}) for s in SPLITS})
        smote_reports, sizes = {}, {s: len(c.X[s]) for s in SPLITS}
        for name in ex["feature_configs"]:
            cols = feats_cfg[name]
            Xs = {s: c.X[s][cols] for s in SPLITS}
            Xsm = ysm = mask = None
            if ex["smote_nc"]:
                sampler = SafeSMOTENC(categorical_columns=[x for x in c.cat if x in cols], k_neighbors=sm["k_neighbors"],
                    target_per_class=sm["target_per_class"], max_ratio=sm["max_ratio"], min_samples=sm["min_samples"],
                    scale_for_neighbours=sm["scale_for_neighbours"], random_state=seed)
                Xsm, ysm = sampler.fit_resample(Xs["train"], c.y["train"])
                mask = np.arange(len(Xsm)) >= sampler.n_original_
                smote_reports[name] = sampler.report_
                assert (Xsm.iloc[: sampler.n_original_].to_numpy() == Xs["train"].to_numpy()).all()
            for rep in ex["representations"]:
                if rep == "tree":
                    export_feature_config(out, "tree", name, Xs, c.y, Xsm, ysm, mask)
                else:
                    enc = MLPEncoder([x for x in c.cat if x in cols], c.prep.categories_).fit(Xs["train"])   # fitted on REAL train rows
                    joblib.dump(enc, art / f"mlp_encoder_{name}.joblib")
                    export_feature_config(out, "mlp", name, {s: enc.transform(Xs[s]) for s in SPLITS}, c.y,
                                          None if Xsm is None else enc.transform(Xsm), ysm, mask)
            log.info("[%s] exported %s (%d features)", ds, name, len(cols))
        joblib.dump(c.prep, art / "tree_preprocessor.joblib")
        joblib.dump(c.le, art / "label_encoder.joblib")
        hmap = c.df.drop_duplicates("label_original").set_index("label_original")["label_harmonized"].to_dict()
        write_json({"classes": {int(i): n for i, n in enumerate(c.le.classes_)}, "n_classes": len(c.le.classes_),
                    "original_to_harmonized": {k: hmap[k] for k in c.le.classes_}, "binary": {"0": "benign", "1": "attack"}}, art / "class_mapping.json")
        write_json(dict(numeric_cols=c.prep.numeric_cols_, categorical_cols=c.prep.categorical_cols_, dropped_constant_in_train=c.prep.dropped_constant_,
                        category_vocabulary=c.prep.categories_, unseen_category_code=-1, feature_sets=feats_cfg,
                        representations={"tree": "median-imputed numeric (unscaled) + ordinal categorical; for RF/XGBoost",
                                         "mlp": "MinMax(clip) numeric + one-hot categorical; for MLP"}), art / "feature_metadata.json")
        write_json(c.sp, art / "split_indices.json")
        names = {str(i): n for i, n in enumerate(c.le.classes_)}
        def _named(rep):
            rep = dict(rep)
            for k in ("before", "after", "skipped_classes_too_small"):
                if k in rep: rep[k] = {names.get(str(a), str(a)): v for a, v in rep[k].items()}
            return rep
        smote_reports = {k: _named(v) for k, v in smote_reports.items()}
        write_json(smote_reports, art / "smote_nc_reports.json")
        write_json(dict(dataset=ds, seed=seed, split_sizes=sizes, python=sys.version, platform=platform.platform(),
                        sklearn=sklearn.__version__, imblearn=imblearn.__version__, pandas=pd.__version__, numpy=np.__version__,
                        config=cfg["smote_nc"]), art / "run_metadata.json")
        write_json(smote_reports, p(cfg, "stats") / f"{ds}_smote_nc_report.json")

        # External binary test set export (e.g. KDDTest+ for NSL-KDD)
        if cfg["datasets"][ds].get("multiclass_source") == "official_train_only" and "__official_split" in c.df.columns:
            ext_rows = c.df[c.df["__official_split"] == "official_test"]
            if not ext_rows.empty:
                ext_dir = out / "external_test_binary"
                ext_dir.mkdir(parents=True, exist_ok=True)
                (ext_dir / "README.md").write_text(
                    "# NSL-KDD External Test Set (Binary Only)\n\n"
                    "This external test set is derived from `KDDTest+.txt`.\n"
                    "It has no subtype labels; labels are binary-only (0 = normal, 1 = attack).\n"
                    "Features are preprocessed using the TRAIN-fitted preprocessor matching the main split pipeline.\n",
                    encoding="utf-8"
                )
                y_ext_bin = ext_rows["label_binary"].to_numpy()
                X_ext_all = c.prep.transform(ext_rows[feature_columns(c.df)]).reset_index(drop=True)
                for name in ex["feature_configs"]:
                    cols = feats_cfg[name]
                    X_ext_tree = X_ext_all[cols]
                    for rep in ex["representations"]:
                        ext_base = ext_dir / rep / name
                        ext_base.mkdir(parents=True, exist_ok=True)
                        if rep == "tree":
                            X_ext_out = X_ext_tree
                        else:
                            enc = joblib.load(art / f"mlp_encoder_{name}.joblib")
                            X_ext_out = enc.transform(X_ext_tree)
                        X_ext_out.to_parquet(ext_base / "X_test.parquet", index=False)
                        pd.DataFrame({"y": y_ext_bin, "y_binary": y_ext_bin}).to_parquet(ext_base / "y_test.parquet", index=False)
                        pd.DataFrame({"y_binary": y_ext_bin}).to_parquet(ext_base / "y_binary.parquet", index=False)
                pd.DataFrame({
                    "row_id": ext_rows["__row_id"].to_numpy(),
                    "label_original": ext_rows["label_original"].to_numpy(),
                    "y_binary": y_ext_bin,
                    "label_harmonized": ext_rows["label_harmonized"].to_numpy()
                }).to_parquet(ext_dir / "labels_test.parquet", index=False)
                log.info("[%s] exported external_test_binary (%d rows) -> %s", ds, len(ext_rows), ext_dir)

        log.info("[%s] export complete -> %s", ds, out)

STEPS = ["inspect", "clean", "split", "select", "export"]
def run(cfg, datasets=None, steps=None):
    avail = available_datasets(cfg)
    datasets = datasets or [d for d, ok in avail.items() if ok]
    missing = [d for d in cfg["datasets"] if not avail.get(d)]
    if missing:
        log.warning("Datasets NOT found (pending): %s", missing)
    if not datasets:
        log.warning("No datasets available - nothing to process. Place the raw files (see README) and re-run. Report will show PENDING.")
        return
    fn = dict(inspect=step_inspect, clean=step_clean, split=step_split, select=step_select, export=step_export)
    for s in (STEPS if steps is None else steps):
        log.info("=== step: %s | datasets: %s", s, datasets)
        fn[s](cfg, datasets)
