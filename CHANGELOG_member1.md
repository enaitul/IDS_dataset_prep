# Member 1 Pipeline Changelog

## 2026-11-15 - Report Tables, NSL-KDD Source Counts & Cleaning Figures

- Updated Section 2 to show KDDTrain+ and KDDTest+ separately using `__official_split`, source counts, kept-feature missing cells, duplicate removal, feature-duplicate rows, and conflicting duplicate groups computed on KDDTrain+ only. The note records that KDDTest+ lacks `difficulty`.
- Corrected the paper-ready NSL-KDD Dataset paragraph to use the multiclass split count and external binary test count without repeating the source count. CICIDS2017 and InSDN validation wording now states that only expected file counts can be checked.
- Added per-class before/after/removal/% removed tables to Section 5 and Figure 9 (PNG/PDF/SVG) for each dataset. Figure 4 now plots missing cells on features retained after cleaning and separates NSL-KDD's official sources.
- Added limitations from the split, SMOTE-NC, and cleaning-effect reports for small training classes, oversampled benign classes, and InSDN DDoS deduplication.
- Added regression checks for report counts, validation wording, class-effect tables, and all requested figures. Only `figures` and `report` pipeline steps were run; raw data, split artifacts, exported data, seed, and SMOTE-NC parameters were not changed.
- Verification: `python -m pytest -q` — 29 passed.

## 2026-11-15 - NSL-KDD Dataset Reporting, InSDN Shortcut Diagnostic & Deliverables

- Updated the report generator to state that NSL-KDD multiclass splitting uses KDDTrain+ only and that KDDTest+ remains a separate binary external test set, with row counts read from inspection, split, and exported label files. Documented `attack` as a coarse binary KDDTest+ label and added a report regression assertion for both generated counts.
- Updated the NSL-KDD `attack` row in `data/metadata/label_mapping.csv` so its mapping reason identifies the coarse binary KDDTest+ source.
- Added `outputs/statistics/InSDN_shortcut_diagnostic.md` using the no-SMOTE tree/all_features exports and TRAIN/VALIDATION only. Recorded RFE and Information Gain feature rankings and Random Forest ablation metrics.
- Added `scripts/package_deliverables.py` to create requested ZIP archives and `deliverables/MANIFEST.md` with archive sizes, file counts, SHA-256 checksums, test result, seed, and library versions. Large model archives are split by feature set when needed.
- Verification: full pytest suite and archive contents are recorded in the generated manifest.
- Methodological safety: raw data, split ratios, seed 42, and SMOTE-NC settings were not changed.

## 2026-11-15 - NSL-KDD Delimiter Auto-Detection

- **Error message:** `ValueError: KDDTrain+.txt: expected 42/43 columns (41 features + label [+ difficulty]), got 1`
- **Root cause:** The real NSL-KDD dataset files (`KDDTrain+.txt` and `KDDTest+.txt`) are tab-delimited (`\t`), whereas `pd.read_csv` defaulted to comma-separated (`sep=','`), causing all columns to be read as a single unparsed text column.
- **Files changed:**
  - `src/data_loader.py`: Added delimiter detection based on reading the first line (`sep = "\t" if "\t" in first_line else ","`) to handle both real tab-separated files and comma-separated synthetic test fixtures with C-engine performance.
  - `tests/test_pipeline.py`: Added `test_nsl_kdd_tab_separated_loading` to verify that tab-separated NSL-KDD files load properly.
- **Methodological safety:** This change only adjusts the CSV parser delimiter to match the actual file format on disk. No data values, rows, labels, or features are altered or removed.

## 2026-11-15 - Handle Empty Core Steps in CLI and Pipeline Runner

- **Error message:** Unexpected re-execution of core pipeline steps (`inspect`) when running `python run_pipeline.py --steps figures report`.
- **Root cause:** When non-core steps (e.g. `figures`, `report`) were passed to `run_pipeline.py`, the core step list became `[]`. In `pipeline.run`, `for s in steps or STEPS:` evaluated `[]` as falsy and defaulted back to all core `STEPS`, triggering an unwanted full pipeline re-run.
- **Files changed:**
  - `src/pipeline.py`: Changed step loop fallback to `for s in (STEPS if steps is None else steps):` so an explicit empty list does not fall back to `STEPS`.
  - `run_pipeline.py`: Guarded `pipeline.run(cfg, a.datasets, core)` with `if core:`.
  - `tests/test_pipeline.py`: Added `test_pipeline_empty_steps_does_not_run_core_steps`.
- **Methodological safety:** Controls execution flow of optional standalone post-processing steps (figures, reports) without re-running data processing. No data, split, or model logic is affected.

## 2026-11-15 - NSL-KDD Multiclass Source Partitioning & Label Case Normalization

- **Methodological Issue:**
  1. `KDDTest+.txt` only contains coarse binary labels (`Normal`: 9,711; `Attack`: 12,833) without specific attack subtypes, whereas `KDDTrain+.txt` provides fine-grained attack subtypes (`neptune`, `satan`, `ipsweep`, etc.). Pooling `KDDTest+` into the main multiclass 70:15:15 split introduced unrefined `Attack` labels into training, validation, and test sets.
  2. Inconsistent label casing (`'normal'` in `KDDTrain+` vs `'Normal'` in `KDDTest+`) caused benign traffic to be partitioned into two distinct classes.
- **Root cause:** NSL-KDD raw data pooling occurred prior to stratified splitting without differentiating the official training source from the coarse test set, and `normalize_label` did not standardize case for NSL-KDD.
- **Files changed:**
  - `configs/config.yaml`: Added `datasets.NSL-KDD.multiclass_source: official_train_only`.
  - `src/label_mapping.py`: Updated `normalize_label(x, dataset)` to lowercase NSL-KDD labels (`'Normal'` $\to$ `'normal'`), unifying benign traffic and eliminating case duplication.
  - `src/cleaning.py`: Passed dataset context to `normalize_label` during inspection and cleaning.
  - `src/dataset_inspection.py`: Lowercased NSL-KDD labels during inspection.
  - `src/pipeline.py`:
    - Updated `step_split`: when `multiclass_source: official_train_only` is configured, only rows from `KDDTrain+` (`__official_split == 'official_train'`) enter the 70:15:15 stratified split.
    - Updated `Context`: verified that all classes present in `train`, `validation`, and `test` splits originate from and exist in `train`.
    - Updated `step_export`: exported `KDDTest+` rows separately as an external binary test set to `data/processed/NSL-KDD/external_test_binary/` with `X` (preprocessed via the TRAIN-fitted preprocessor across all representations and feature configs), `y_binary`, and an informative `README.md`.
  - `src/export_data.py`: Added `load_external_test()` utility function.
  - `README.md`: Added documentation for `external_test_binary/`.
  - `tests/test_pipeline.py`: Added tests:
    - `test_nsl_kdd_test_rows_not_in_split`: verifies 0 `KDDTest+` rows enter any split.
    - `test_nsl_kdd_external_test_same_column_order`: verifies external test set column alignment with `X_test`.
    - `test_nsl_kdd_label_case_duplicates_gone`: verifies `'Normal'` and `'normal'` are merged and no case duplicates remain.
- **Methodological safety:** Prevents coarse, un-subtyped labels from corrupting multiclass training and evaluation; adheres strictly to the 70:15:15 ratio and random seed (42) on the legitimate multiclass training corpus; keeps KDDTest+ completely isolated as an out-of-sample external binary evaluation set transformed only by train-fitted statistics.
