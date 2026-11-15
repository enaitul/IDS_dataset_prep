# SDN_IDS - Member 1: Dataset Preparation & Feature Engineering

Reproducible, leakage-controlled data pipeline for *"An Enhanced Intrusion Detection System for SDN Using Hybrid Neural Model and Ensemble Learning"* (CICIDS2017, InSDN, NSL-KDD).

> **Status of this package:** the code was verified with automated tests on small TEST-ONLY synthetic fixtures that mimic each dataset's schema quirks. **It has not been run on the real datasets yet** - there are therefore no real statistics, selected features or figures in this package. Running it on your machine produces them.

## 1. Put the raw data in place (originals, unmodified)

```
data/raw/NSL-KDD/     KDDTrain+.txt  KDDTest+.txt
data/raw/CICIDS2017/  GeneratedLabelledFlows.zip   (or the extracted 8 CSVs, any sub-folder)
data/raw/InSDN/       InSDN_DatasetCSV.zip         (or Normal_data.csv, OVS.csv, metasploitable-2.csv)
```
Zip files are extracted automatically. Check with `python check_raw.py` (prints size, MD5, rows; NSL-KDD is compared with published row counts and the Zenodo MD5s).

## 2. Run

```
pip install -r requirements.txt
python run_pipeline.py                       # all available datasets, all steps
python run_pipeline.py --datasets NSL-KDD    # one dataset (fastest; start here)
python run_pipeline.py --steps select export # re-run later steps only
python -m pytest tests -q                    # verification suite
```
Steps: `inspect -> clean -> split -> select -> export -> figures -> report`. Each reads the previous step's files, so they can be re-run individually. Datasets that are missing are skipped with a warning and marked PENDING in the report.

**Resources (CICIDS2017):** ~2.8 M raw flows - use a machine with >= 16 GB RAM. Information gain and RFE run on stratified subsamples (config). Expect minutes for NSL-KDD/InSDN and substantially longer for CICIDS2017 (RFE + SMOTE-NC per feature configuration).

## 3. Outputs

| Path | Content |
|---|---|
| `data/processed/cleaned/<ds>.parquet` | cleaned data with `label_original`, `label_harmonized`, `label_binary`, `__row_id` |
| `data/splits/<ds>_split_indices.json` | exact `__row_id`s per split + seed + metadata (+ official NSL-KDD split ids) |
| `data/processed/<ds>/{tree,mlp}/<feature_set>/` | `X_/y_{train,validation,test}.parquet`, plus `X_train_smotenc.parquet`/`y_train_smotenc.parquet` (`is_synthetic` flag, train only) |
| `data/processed/NSL-KDD/external_test_binary/{tree,mlp}/<set>/` | External test set derived from KDDTest+ (TRAIN-preprocessed; binary labels only, no subtype labels) |
| `data/processed/<ds>/labels/` | all label views per split |
| `data/processed/<ds>/artifacts/` | `tree_preprocessor.joblib`, `mlp_encoder_<set>.joblib`, `label_encoder.joblib`, `class_mapping.json`, `feature_metadata.json`, `split_indices.json`, `smote_nc_reports.json`, `run_metadata.json` |
| `data/metadata/label_mapping.csv` | dataset, original label, harmonised label, confidence, reason |
| `outputs/statistics/` | summary / class distribution / missing / duplicate / schema / cleaning / split / leakage-audit / SMOTE reports |
| `outputs/selected_features/` | IG ranking, RFE ranking, IG->RFE ranking, k-tuning curves, selected feature JSON |
| `outputs/graphs/` | figures (PNG 300 dpi + PDF + SVG) |
| `outputs/reports/Member1_Dataset_and_Feature_Engineering_Report.md` | report + paper-ready text (generated from the files above) |

Feature sets: `all_features`, `info_gain`, `rfe`, `info_gain_rfe`. Representations: `tree` (median-imputed, **unscaled**, ordinal categoricals; for RF/XGBoost) and `mlp` (Min-Max clipped + one-hot; for MLP).

## 4. For Members 2 and 3

```python
from src.export_data import load_export, load_cleaned_split
d = load_export("data/processed", "NSL-KDD", representation="tree", features="rfe", smote=True)
d["X_train"], d["y_train"], d["X_validation"], d["y_validation"], d["X_test"], d["y_test"]
```
`smote=True` gives the SMOTE-NC-resampled **training** set (validation/test are never resampled). Use it for plain fits.
**For cross-validation or out-of-fold stacking do not reuse that array.** Use `load_cleaned_split()` + `src/model_ready.build_pipeline()` / `oof_probabilities()`, which refit imputation, encoding, selection (optional) and SMOTE-NC inside every training fold. Wrap XGBoost in `ContiguousLabelClassifier` (rare classes can be absent from a fold). See `examples/consume_exports.py`. Keep the **test** set untouched until final evaluation. Rare classes that SMOTE-NC refuses to oversample are listed in `smote_nc_reports.json` - use class weights for them.

## 5. Key design decisions (all configurable in `configs/config.yaml`)
* Split first; every fitted object uses training data only; validation labels are used only to choose the RFE feature count.
* Exact duplicates (features + label) removed, first occurrence kept; conflicting duplicates (same features, different labels) are kept and reported.
* Identifier columns removed only if listed in `identifier_columns` with a reason; `Timestamp`/`difficulty` are kept as `__` metadata for audits.
* `Destination Port` is kept (possible shortcut feature - run an ablation).
* SMOTE-NC is capped (never above the majority class, at most `max_ratio` x real size, none for classes with < `min_samples` rows).
* Random stratified splitting of flows can be optimistic; the leakage audit quantifies cross-split near-duplicates.

See `outputs/reports/Spec_Conflicts.md` for the manuscript-vs-brief conflicts that need a team decision.
