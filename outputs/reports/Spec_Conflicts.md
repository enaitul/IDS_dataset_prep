# Specification conflicts found (manuscript "Research Paper latest" vs. Member 1 brief)

Status: **decisions below were taken as defaults; please confirm or overrule.** Nothing here is irreversible - every choice is a config value or a documented export.

| # | Manuscript says | Member 1 brief says | What this pipeline does | Needs team decision? |
|---|---|---|---|---|
| 1 | Hybrid CNN-RNN/LSTM + voting/stacking | MLP + RF + XGBoost, Logistic-Regression meta-classifier | Targets the brief. Exports 2-D tabular data (tree and mlp representations); a CNN/LSTM can reshape the `mlp` arrays. No architecture was silently swapped. | **Yes** - the paper (Sec. IV-E, Algorithm Steps 6-7, intro) or the implementation must be aligned. |
| 2 | SMOTE | SMOTE-NC | SMOTE-NC. It reduces to plain SMOTE when no categorical column survives (e.g. InSDN/CICIDS2017 after selection if `Protocol` is dropped); this is recorded per run. | Wording in paper. |
| 3 | Algorithm: Step 2 normalise -> Step 4 SMOTE -> Step 5 split | Split first, fit everything on train only | Split first. Scaling/imputation/selection/SMOTE-NC all fitted on training data only; SMOTE-NC re-run inside each CV fold. The paper's step order would leak test information and put synthetic rows in the test set. | **Yes** - correct the Algorithm section. |
| 4 | Train/test split | 70:15:15 | 70:15:15 with a validation set. | Update Sec. IV-B. |
| 5 | Min-Max scaling of all numerics | Do not unnecessarily scale for tree models | `tree` export is unscaled; `mlp` export is Min-Max (clipped) + one-hot. | No. |
| 6 | RFE and Information Gain "such as" | Implement both | Exported separately (`info_gain`, `rfe`) and as a hybrid (`info_gain_rfe`: IG filter then RFE) so Members 2/3 can compare. | Which configuration is "the" proposed one? |
| 7 | Output: Normal/Attack + category | - | Both `y_binary` and multiclass `y` exported (plus harmonised labels). | No. |
| 8 | Datasets "reflect realistic SDN traffic" | - | Only InSDN was captured in an SDN environment; CICIDS2017 and NSL-KDD were not. Reported under limitations. | Paper wording. |
| 9 | Generalisation across datasets | Harmonised labels only for carefully defined cross-dataset experiments | Per-dataset experiments are the default; `label_harmonized` is exported for cross-dataset runs, UNMAPPED labels excluded there. | Confirm the experiment design. |
| 10 | NSL-KDD | - | KDDTrain+ and KDDTest+ are pooled then re-split 70:15:15; official split row-ids are stored (`official_train_row_ids`, `official_test_row_ids`) for a secondary benchmark. | Confirm pooling. |
