"""Verification suite. Uses TEST-ONLY synthetic fixtures; no result here is a finding about the real datasets."""
import copy, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src import pipeline
from src.data_loader import load_dataset, DatasetNotFound
from src.utils import load_config, read_json, feature_columns
from src.label_mapping import normalize_label, harmonize, is_benign, build_mapping_table, UNMAPPED
from src.splitting import stratified_split_indices
from src.preprocessing import TreePreprocessor, MLPEncoder, ColumnSelector
from src.smote import SafeSMOTENC
from src.export_data import load_export
from src.model_ready import build_pipeline, oof_probabilities, ContiguousLabelClassifier
from tests.fixtures import make_workspace

DS = ["NSL-KDD", "CICIDS2017", "InSDN"]

# ------------------------------------------------------------------ loading / inspection
def test_loading_schemas(ws):
    tmp, cfg = ws
    nsl, i1 = load_dataset("NSL-KDD", cfg); cic, i2 = load_dataset("CICIDS2017", cfg); ins, i3 = load_dataset("InSDN", cfg)
    assert nsl.shape[1] == 43 + 3 and "label" in nsl and "__official_split" in nsl          # 43 + __row_id/__source_file/__official_split
    assert "Label" in cic.columns and "Destination Port" in cic.columns                      # leading spaces stripped
    assert len(i3) == 3 and ins["__source_file"].nunique() == 3
    assert any("\u2013" in str(x) or "XSS" in str(x) for x in cic["Label"].dropna().unique())
    assert i1[0]["validation"]["rows_match_published"] is False                              # fixture is not the real NSL-KDD: validator must say so

def test_missing_dataset_raises(tmp_path):
    cfg = make_workspace(tmp_path, datasets=())
    with pytest.raises(DatasetNotFound):
        load_dataset("InSDN", cfg)
    assert pipeline.available_datasets(cfg) == {"NSL-KDD": False, "CICIDS2017": False, "InSDN": False}

def test_nsl_kdd_tab_separated_loading(tmp_path):
    cfg = make_workspace(tmp_path, datasets=["NSL-KDD"])
    nsl_dir = Path(cfg["_root"]) / cfg["datasets"]["NSL-KDD"]["raw_dir"]
    # Rewrite the synthetic files in the isolated tmp workspace as tab-separated
    for f in ["KDDTrain+.txt", "KDDTest+.txt"]:
        p = nsl_dir / f
        df = pd.read_csv(p, header=None)
        df.to_csv(p, sep="\t", header=False, index=False)
    df, infos = load_dataset("NSL-KDD", cfg)
    assert df.shape[1] == 43 + 3 and "label" in df and "__official_split" in df

def test_inspection_counts(ws):
    _, cfg = ws
    r = read_json(Path(cfg["_root"]) / "outputs/statistics/CICIDS2017_inspection.json")["summary"]
    assert r["inf_cells"] == 2 and r["rows_without_label"] == 3
    n = read_json(Path(cfg["_root"]) / "outputs/statistics/NSL-KDD_inspection.json")["summary"]
    assert n["exact_duplicate_rows"] == 25 and n["n_constant_features"] >= 1
    for f in ["dataset_summary", "class_distribution", "missing_value_report", "duplicate_report", "feature_schema_report"]:
        assert (Path(cfg["_root"]) / f"outputs/statistics/{f}.csv").exists()

# ------------------------------------------------------------------ cleaning
def test_duplicate_removal_and_documented_columns(ws):
    _, cfg = ws
    for ds, expected_dups in [("NSL-KDD", 25), ("CICIDS2017", 15)]:
        df = pd.read_parquet(pipeline.cleaned_path(cfg, ds))
        assert not df.duplicated(subset=feature_columns(df) + ["label_original"]).any()
        assert read_json(Path(cfg["_root"]) / f"outputs/statistics/{ds}_cleaning_report.json")["rows_removed"]["exact_duplicates"] == expected_dups
    rem = pd.read_csv(Path(cfg["_root"]) / "outputs/statistics/CICIDS2017_removed_columns.csv")
    assert set(["Flow ID", "Source IP", "Timestamp", "Const Col", "Fwd Header Length.1"]) <= set(rem["column"])
    assert rem["reason"].str.len().min() > 10
    assert "__timestamp" in pd.read_parquet(pipeline.cleaned_path(cfg, "CICIDS2017")).columns

def test_missing_and_inf_handling(ws):
    _, cfg = ws
    df = pd.read_parquet(pipeline.cleaned_path(cfg, "CICIDS2017"))
    X = df[feature_columns(df)].select_dtypes("number")
    assert not np.isinf(X.to_numpy()).any()
    assert df[feature_columns(df)].isna().sum().sum() == 3     # 1 NaN + 2 former infinities, NOT imputed during cleaning
    assert df["label_original"].notna().all() and "nan" not in set(df["label_original"])

# ------------------------------------------------------------------ labels
def test_label_mapping():
    assert normalize_label("Web Attack \u2013 Brute Force") == "Web Attack - Brute Force"
    assert normalize_label("Web Attack \x96 XSS") == "Web Attack - XSS"
    assert harmonize("CICIDS2017", "Web Attack - XSS") == "WebAttack" and harmonize("InSDN", "BOTNET") == "Botnet"
    assert harmonize("NSL-KDD", "neptune") == "DoS" and harmonize("NSL-KDD", "ipsweep") == "Probe"
    assert harmonize("NSL-KDD", "rootkit") == UNMAPPED and harmonize("CICIDS2017", "Heartbleed") == UNMAPPED   # not forced
    assert harmonize("InSDN", "totally_new_attack") == UNMAPPED
    assert is_benign("NSL-KDD", "normal") and is_benign("CICIDS2017", "BENIGN") and is_benign("InSDN", "Normal") and not is_benign("InSDN", "DoS")
    t = build_mapping_table({"InSDN": ["Normal", "weird"]})
    assert list(t.columns[:4]) == ["dataset", "original_label", "harmonized_label", "confidence"]
    assert t.loc[t.original_label == "weird", "confidence"].iloc[0] == "unknown"

def test_label_columns_consistent(ws):
    _, cfg = ws
    for ds in DS:
        df = pd.read_parquet(pipeline.cleaned_path(cfg, ds))
        assert ((df["label_harmonized"] == "Benign") == (df["label_binary"] == 0)).all()

# ------------------------------------------------------------------ splitting
def test_split_ratios_disjoint_and_all_classes_in_train(ws):
    _, cfg = ws
    for ds in DS:
        sp = read_json(pipeline.split_path(cfg, ds)); df = pd.read_parquet(pipeline.cleaned_path(cfg, ds))
        tr, va, te = (set(sp[k]) for k in ("train", "validation", "test"))
        assert not (tr & va) and not (tr & te) and not (va & te)
        if ds == "NSL-KDD" and cfg["datasets"][ds].get("multiclass_source") == "official_train_only" and "__official_split" in df.columns:
            train_rids = set(df[df["__official_split"] == "official_train"]["__row_id"])
            assert tr | va | te == train_rids
            assert not (tr & set(sp.get("official_test_row_ids", [])))
            assert not (va & set(sp.get("official_test_row_ids", [])))
            assert not (te & set(sp.get("official_test_row_ids", [])))
            n = len(train_rids)
            lab = df[df["__official_split"] == "official_train"].set_index("__row_id")["label_original"]
        else:
            assert tr | va | te == set(df["__row_id"])
            n = len(df)
            lab = df.set_index("__row_id")["label_original"]
        assert abs(len(tr) / n - .70) < .02 and abs(len(va) / n - .15) < .02 and abs(len(te) / n - .15) < .02
        assert set(lab.loc[list(tr)]) == set(lab)            # every class present in train
        assert sp["metadata"]["seed"] == cfg["seed"]

def test_small_class_handling():
    y = np.array(["a"] * 100 + ["b"] * 5 + ["c"] * 2 + ["d"] * 1)
    idx, meta = stratified_split_indices(y, 1)
    sizes = {c: [int((y[idx[k]] == c).sum()) for k in ("train", "validation", "test")] for c in "abcd"}
    assert sizes["a"] == [70, 15, 15] and sum(sizes["b"]) == 5 and min(sizes["b"]) >= 1
    assert sizes["c"] == [1, 0, 1] and sizes["d"] == [1, 0, 0]
    assert {t["label"] for t in meta["classes_too_small_for_exact_stratification"]} == {"b", "c", "d"}

def test_reproducible_split():
    y = np.random.default_rng(0).choice(list("abcde"), 500, p=[.5, .2, .15, .1, .05])
    a, _ = stratified_split_indices(y, 7); b, _ = stratified_split_indices(y, 7); c, _ = stratified_split_indices(y, 8)
    assert all((a[k] == b[k]).all() for k in a) and not (a["train"].shape == c["train"].shape and (a["train"] == c["train"]).all())

def test_no_exact_duplicates_across_splits(ws):
    _, cfg = ws
    for ds in DS:
        a = read_json(Path(cfg["_root"]) / f"outputs/statistics/{ds}_leakage_audit.json")
        assert a["exact_duplicate_rows_test_in_train"] == 0 and a["exact_duplicate_rows_validation_in_train"] == 0 and a["exact_duplicate_rows_test_in_validation"] == 0

# ------------------------------------------------------------------ preprocessing / leakage
def test_preprocessor_fitted_on_train_only(ws):
    _, cfg = ws
    c = pipeline.Context(cfg, "NSL-KDD")
    tr = c.rows["train"][feature_columns(c.df)]
    num = [x for x in c.prep.numeric_cols_]
    assert np.allclose(pd.Series(c.prep.medians_)[num], tr[num].median()[num])
    allmed = c.df[feature_columns(c.df)][num].median()
    assert not np.allclose(pd.Series(c.prep.medians_)[num], allmed[num])          # would coincide only if test data were used
    # transforming wildly different data must not change fitted statistics
    wild = c.rows["test"][feature_columns(c.df)].copy(); wild[num] = wild[num] * 1e6
    before = dict(c.prep.medians_); c.prep.transform(wild)
    assert before == c.prep.medians_

def test_unknown_categories_safe(ws):
    _, cfg = ws
    c = pipeline.Context(cfg, "NSL-KDD")
    row = c.rows["test"][feature_columns(c.df)].head(3).copy()
    row["service"] = "never_seen_service"; row.loc[row.index[0], "flag"] = np.nan
    out = c.prep.transform(row)
    assert (out["service"] == -1).all() and out["flag"].notna().all()
    enc = MLPEncoder(c.cat, c.prep.categories_).fit(c.X["train"])
    z = enc.transform(out)
    assert z[[n for n in z.columns if n.startswith("service__")]].to_numpy().sum() == 0
    assert z.shape[1] == len(enc.feature_names_out_) and z.notna().all().all()

# ------------------------------------------------------------------ SMOTE-NC
def test_smote_only_on_train_exports(ws):
    tmp, cfg = ws
    for ds in DS:
        out = Path(cfg["_root"]) / "data/processed" / ds
        c = pipeline.Context(cfg, ds)
        for rep in ("tree", "mlp"):
            d = load_export(out.parent, ds, rep, "rfe", smote=True)
            n_tr = len(c.X["train"])
            assert len(d["X_validation"]) == len(c.X["validation"]) and len(d["X_test"]) == len(c.X["test"])      # untouched
            assert len(d["X_train"]) > n_tr and d["is_synthetic"].sum() == len(d["X_train"]) - n_tr
            assert not d["is_synthetic"][:n_tr].any() and d["is_synthetic"][n_tr:].all()
            plain = load_export(out.parent, ds, rep, "rfe", smote=False)
            assert len(plain["X_train"]) == n_tr
            assert np.array_equal(d["X_train"].iloc[:n_tr].to_numpy(), plain["X_train"].to_numpy())             # originals bit-identical
            # validation/test must equal an INDEPENDENT recomputation from the split (not merely another read of the same file)
            cols = list(plain["X_validation"].columns)
            ref = {s_: c.X[s_][[x for x in cols if x in c.X[s_].columns]] for s_ in ("validation", "test")} if rep == "tree" else None
            if rep == "tree":
                for s_ in ("validation", "test"):
                    assert np.array_equal(d[f"X_{s_}"].to_numpy(), ref[s_].to_numpy()) and np.array_equal(d[f"y_{s_}"], c.y[s_])
            else:
                import joblib
                enc = joblib.load(out / "artifacts" / "mlp_encoder_rfe.joblib")
                sel_cols = read_json(out / "artifacts" / "feature_metadata.json")["feature_sets"]["rfe"]
                for s_ in ("validation", "test"):
                    assert np.allclose(d[f"X_{s_}"].to_numpy(), enc.transform(c.X[s_][sel_cols]).to_numpy()) and np.array_equal(d[f"y_{s_}"], c.y[s_])
        rep_ = read_json(Path(cfg["_root"]) / f"outputs/statistics/{ds}_smote_nc_report.json")["rfe"]
        assert sum(rep_["after"].values()) == len(load_export(out.parent, ds, "tree", "rfe", smote=True)["X_train"])
        assert all(rep_["after"][k] >= v for k, v in rep_["before"].items())

def test_smote_skips_tiny_classes_and_caps():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"n1": rng.normal(size=330), "n2": rng.normal(size=330), "cat": rng.integers(0, 3, 330).astype(float)})
    y = np.array([0] * 300 + [1] * 25 + [2] * 3 + [3] * 2)
    s = SafeSMOTENC(categorical_columns=["cat"], target_per_class=200, max_ratio=4, min_samples=6, random_state=0)
    Xr, yr = s.fit_resample(X, y)
    cnt = pd.Series(yr).value_counts()
    assert cnt[1] == 100 and cnt[2] == 3 and cnt[3] == 2 and cnt[0] == 300         # capped at 4x; tiny classes untouched
    assert s.report_["skipped_classes_too_small"] == {"2": 3, "3": 2}
    assert np.array_equal(Xr.iloc[:330].to_numpy(), X.to_numpy())
    assert set(Xr["cat"].unique()) <= {0.0, 1.0, 2.0}                               # categorical values stay valid categories

def test_smote_without_categorical_falls_back_to_smote():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.normal(size=(120, 3)), columns=list("abc")); y = np.array([0] * 100 + [1] * 20)
    s = SafeSMOTENC(categorical_columns=[], target_per_class=100, max_ratio=10, random_state=0); _, yr = s.fit_resample(X, y)
    assert s.report_["method"].startswith("SMOTE (no categorical") and (yr == 1).sum() == 100

def test_smote_inside_cv_folds_only_on_training_part(ws, monkeypatch):
    _, cfg = ws
    c = pipeline.Context(cfg, "NSL-KDD")
    Xdf = pd.concat([c.rows["train"]]).reset_index(drop=True)
    y = c.le.transform(Xdf["label_original"]); X = Xdf[feature_columns(c.df)]
    seen = []
    orig = SafeSMOTENC.fit_resample
    def spy(self, X_, y_):
        seen.append(len(X_)); return orig(self, X_, y_)
    monkeypatch.setattr(SafeSMOTENC, "fit_resample", spy)
    smote_cfg = cfg["smote_nc"]
    models = {"rf": build_pipeline(RandomForestClassifier(20, random_state=0), c.spec, None, "tree", smote_cfg, seed=0),
              "xgb": build_pipeline(ContiguousLabelClassifier(XGBClassifier(n_estimators=10, max_depth=3, verbosity=0)), c.spec, None, "tree", smote_cfg, seed=0)}
    oof = oof_probabilities(models, X, y, np.arange(len(c.le.classes_)), n_splits=3, seed=0)
    assert len(seen) == 6                                          # 3 folds x 2 models: resampling happened per fold ...
    assert all(s < len(X) for s in seen)                            # ... on a strict subset, never on the whole training set
    assert sorted(set(seen)) and max(seen) <= len(X) - len(X) // 3 + 3
    for name, P in oof.items():
        assert P.shape == (len(X), len(c.le.classes_)) and np.allclose(P.sum(axis=1), 1, atol=1e-6)

# ------------------------------------------------------------------ feature selection / export
def test_selected_features_consistent_and_ordered(ws):
    _, cfg = ws
    for ds in DS:
        sel = read_json(Path(cfg["_root"]) / f"outputs/selected_features/{ds}_selected_features.json")
        meta = read_json(Path(cfg["_root"]) / f"data/processed/{ds}/artifacts/feature_metadata.json")
        for name in ("all_features", "info_gain", "rfe", "info_gain_rfe"):
            cols = meta["feature_sets"][name]
            for rep in ("tree",):
                d = load_export(Path(cfg["_root"]) / "data/processed", ds, rep, name, smote=False)
                ds_ = load_export(Path(cfg["_root"]) / "data/processed", ds, rep, name, smote=True)
                assert list(d["X_train"].columns) == list(d["X_validation"].columns) == list(d["X_test"].columns) == list(ds_["X_train"].columns) == cols
        assert set(sel["info_gain_rfe"]["features"]) <= set(sel["info_gain"]["features"])          # hybrid restricted to IG-retained set
        mlp = load_export(Path(cfg["_root"]) / "data/processed", ds, "mlp", "rfe")
        assert list(mlp["X_train"].columns) == list(mlp["X_test"].columns)
        assert mlp["X_train"].to_numpy().min() >= 0 and mlp["X_test"].to_numpy().max() <= 1 + 1e-9        # MinMax clip + one-hot
    s = ColumnSelector(["b", "a"]).fit(pd.DataFrame({"a": [1], "b": [2], "c": [3]}))
    assert list(s.transform(pd.DataFrame({"c": [0], "a": [1], "b": [2]})).columns) == ["b", "a"]

def test_selection_uses_no_validation_labels(ws, tmp_path):
    """Shuffling VALIDATION/TEST labels must not change the IG or RFE rankings (only the RFE count may change)."""
    _, cfg = ws
    cfg2 = make_workspace(tmp_path, datasets=())
    shutil.copytree(Path(cfg["_root"]) / "data", tmp_path / "data", dirs_exist_ok=True)
    cfg2 = load_config(tmp_path / "configs" / "config.yaml")
    ds = "InSDN"; f = pipeline.cleaned_path(cfg2, ds); df = pd.read_parquet(f)
    sp = read_json(pipeline.split_path(cfg2, ds)); mask = df["__row_id"].isin(sp["validation"] + sp["test"])
    df.loc[mask, "label_original"] = np.random.default_rng(0).permutation(df.loc[mask, "label_original"].to_numpy())
    df.to_parquet(f, index=False)
    pipeline.step_select(cfg2, [ds])
    a = pd.read_csv(Path(cfg["_root"]) / f"outputs/selected_features/{ds}_info_gain_ranking.csv"); b = pd.read_csv(tmp_path / f"outputs/selected_features/{ds}_info_gain_ranking.csv")
    assert a["feature"].tolist() == b["feature"].tolist() and np.allclose(a["info_gain"], b["info_gain"])
    ra = pd.read_csv(Path(cfg["_root"]) / f"outputs/selected_features/{ds}_rfe_ranking.csv"); rb = pd.read_csv(tmp_path / f"outputs/selected_features/{ds}_rfe_ranking.csv")
    assert ra["feature"].tolist() == rb["feature"].tolist()

def test_export_reload_correctness(ws):
    _, cfg = ws
    root = Path(cfg["_root"]) / "data/processed"
    for ds in DS:
        d = load_export(root, ds, "tree", "all_features")
        c = pipeline.Context(cfg, ds)
        assert np.allclose(d["X_train"].to_numpy(), c.X["train"].to_numpy()) and np.array_equal(d["y_train"], c.y["train"])
        assert np.allclose(d["X_test"].to_numpy(), c.X["test"].to_numpy()) and np.array_equal(d["y_test"], c.y["test"])
        inv = d["label_encoder"].inverse_transform(d["y_test"]); assert (inv == c.rows["test"]["label_original"].to_numpy()).all()
        assert set(d["class_mapping"]["classes"].values()) == set(d["label_encoder"].classes_)
        re = d["preprocessor"].transform(c.rows["test"][feature_columns(c.df)]); assert np.allclose(re.to_numpy(), d["X_test"].to_numpy())
        lab = d["labels"]["test"]; assert list(lab.columns) == ["row_id", "label_original", "y_multiclass", "y_binary", "label_harmonized"]
        assert set(lab["row_id"]) == set(read_json(pipeline.split_path(cfg, ds))["test"])
    with pytest.raises(FileNotFoundError):
        load_export(root, "NSL-KDD", "tree", "nonexistent")

def test_repeated_run_is_reproducible(ws, tmp_path):
    _, cfg = ws
    cfg2 = make_workspace(tmp_path, datasets=["InSDN"])
    pipeline.run(cfg2, ["InSDN"])
    a = read_json(pipeline.split_path(cfg, "InSDN")); b = read_json(pipeline.split_path(cfg2, "InSDN"))
    assert a["train"] == b["train"] and a["validation"] == b["validation"] and a["test"] == b["test"]
    for cfgname in ("all_features", "rfe"):
        for rep in ("tree", "mlp"):
            x1 = load_export(Path(cfg["_root"]) / "data/processed", "InSDN", rep, cfgname, smote=True)
            x2 = load_export(Path(cfg2["_root"]) / "data/processed", "InSDN", rep, cfgname, smote=True)
            assert x1["X_train"].equals(x2["X_train"]) and np.array_equal(x1["y_train"], x2["y_train"])
    i1 = pd.read_csv(Path(cfg["_root"]) / "outputs/selected_features/InSDN_info_gain_ranking.csv"); i2 = pd.read_csv(Path(cfg2["_root"]) / "outputs/selected_features/InSDN_info_gain_ranking.csv")
    assert i1["feature"].tolist() == i2["feature"].tolist()
    assert read_json(Path(cfg["_root"]) / "outputs/selected_features/InSDN_selected_features.json")["rfe"]["features"] == \
           read_json(Path(cfg2["_root"]) / "outputs/selected_features/InSDN_selected_features.json")["rfe"]["features"]

def test_figures_and_report_generated(ws):
    _, cfg = ws
    from src import visualization, reporting
    visualization.make_all(cfg); rp = reporting.write_report(cfg)
    g = Path(cfg["_root"]) / "outputs/graphs"
    for stem in ["fig01_class_distribution_CICIDS2017", "fig02_class_distribution_InSDN", "fig03_class_distribution_NSL-KDD", "fig04_missing_value_comparison",
                 "fig05_smote_before_after_NSL-KDD", "fig06_information_gain_ranking_InSDN", "fig07_rfe_ranking_InSDN", "fig08_feature_count_comparison"]:
        for ext in ("png", "pdf", "svg"):
            assert (g / f"{stem}.{ext}").exists(), stem
    for ds in DS:
        for ext in ("png", "pdf", "svg"):
            assert (g / f"fig09_cleaning_class_effect_{ds}.{ext}").exists()
    txt = rp.read_text(encoding="utf-8")
    assert "Research-paper-ready text" in txt and "## 12. Experimental parameters" in txt
    assert "PENDING" not in txt.split("## 1. Dataset descriptions")[1]          # all three fixture datasets were processed
    nsl_inspection = read_json(Path(cfg["_root"]) / "outputs/statistics/NSL-KDD_inspection.json")
    split_meta = read_json(Path(cfg["_root"]) / "data/splits/NSL-KDD_split_indices.json")["metadata"]
    train_rows = sum(split_meta["sizes"].values())
    external_rows = len(pd.read_parquet(Path(cfg["_root"]) / "data/processed/NSL-KDD/external_test_binary/tree/all_features/y_test.parquet"))
    assert f"KDDTrain+ only ({train_rows:,} rows" in txt
    assert f"KDDTest+ is a separate external binary test set ({external_rows:,} rows after duplicate removal" in txt
    assert f"multiclass KDDTrain+ split: {train_rows:,} rows" in txt
    assert f"external binary KDDTest+: {external_rows:,} rows after duplicate removal" in txt
    assert "125,973 rows; 125,973 source rows" not in txt
    cleaned_nsl = pd.read_parquet(Path(cfg["_root"]) / "data/processed/cleaned/NSL-KDD.parquet")
    train_nsl = cleaned_nsl[cleaned_nsl["__official_split"] == "official_train"]
    kept = feature_columns(cleaned_nsl)
    train_missing = int(train_nsl[kept].isna().sum().sum())
    train_source_rows = next(x["rows"] for x in nsl_inspection["file_infos"] if x["file"] == "KDDTrain+.txt")
    assert f"KDDTrain+ | {train_source_rows:,}" in txt and f"{train_missing:,}" in txt
    assert "KDDTest+" in txt and "Conflicting duplicate groups" in txt
    for ds in ("CICIDS2017", "InSDN"):
        assert f"{ds}" in txt
    assert "file count matches expected; no published row-count/checksum reference available" in txt
    assert "Post-cleaning class effect" in txt
    assert "Duplicates are identified after removing identifier columns" in txt

def test_cleaned_split_loader_matches_exports(ws):
    from src.export_data import load_cleaned_split
    _, cfg = ws
    for ds in DS:
        r = load_cleaned_split(cfg, ds); e = load_export(Path(cfg["_root"]) / "data/processed", ds, "tree", "all_features")
        assert len(r["X_train_raw"]) == len(e["X_train"]) and np.array_equal(r["y_val" if False else "y_validation"], e["y_validation"])
        assert np.array_equal(r["y_test"], e["y_test"])

def test_no_datasets_is_pending_not_crash(tmp_path):
    """With no raw data the pipeline must not crash, must not fabricate outputs, and the report must say PENDING."""
    from src import reporting
    cfg = make_workspace(tmp_path, datasets=())
    pipeline.run(cfg)
    rp = reporting.write_report(cfg)
    txt = rp.read_text(encoding="utf-8")
    assert "Datasets processed end-to-end: **none**" in txt and "PENDING" in txt
    assert not list((tmp_path / "outputs/statistics").glob("*.csv")) and not list((tmp_path / "data/processed").rglob("*.parquet"))

def test_pipeline_empty_steps_does_not_run_core_steps(tmp_path):
    cfg = make_workspace(tmp_path, datasets=["NSL-KDD"])
    pipeline.run(cfg, datasets=["NSL-KDD"], steps=[])
    # With steps=[], no inspection/cleaning/splitting/etc. should have run
    assert not list((tmp_path / "data/processed").rglob("*.parquet"))

def test_nsl_kdd_test_rows_not_in_split(ws):
    """(a) No KDDTest+ row_id appears in any train/validation/test split."""
    _, cfg = ws
    sp = read_json(pipeline.split_path(cfg, "NSL-KDD"))
    test_rids = set(sp.get("official_test_row_ids", []))
    assert test_rids, "official_test_row_ids must be present"
    for s in ("train", "validation", "test"):
        overlap = set(sp[s]) & test_rids
        assert not overlap, f"KDDTest+ row_ids leaked into {s} split: {len(overlap)} rows"

def test_nsl_kdd_external_test_same_column_order(ws):
    """(b) The external test set is exported with the exact same column order as X_test."""
    from src.export_data import load_external_test, load_export
    _, cfg = ws
    root = Path(cfg["_root"]) / "data/processed"
    for rep in ("tree", "mlp"):
        for feat in ("all_features", "info_gain", "rfe", "info_gain_rfe"):
            main_exp = load_export(root, "NSL-KDD", representation=rep, features=feat)
            ext_exp = load_external_test(root, "NSL-KDD", representation=rep, features=feat)
            assert list(ext_exp["X_test"].columns) == list(main_exp["X_test"].columns)
            assert len(ext_exp["X_test"]) == len(ext_exp["y_test"])
            assert set(ext_exp["y_test"]).issubset({0, 1})

def test_nsl_kdd_label_case_duplicates_gone(ws):
    """(c) 'Normal' and 'normal' are unified into 'normal' (no case duplicate classes)."""
    _, cfg = ws
    df = pd.read_parquet(pipeline.cleaned_path(cfg, "NSL-KDD"))
    classes = set(df["label_original"])
    assert "Normal" not in classes
    assert "normal" in classes
    assert len(classes) == len({c.lower() for c in classes})
