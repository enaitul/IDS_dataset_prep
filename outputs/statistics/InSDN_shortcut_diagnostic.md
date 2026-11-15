# InSDN Shortcut Diagnostic

Read-only diagnostic using the saved TRAIN and VALIDATION exports only. The external test split was not loaded or evaluated. Model: `RandomForestClassifier(n_estimators=100, random_state=42)`; representation: tree, features: all_features, SMOTE: disabled.

## RFE-selected features

- `Init Bwd Win Byts`
- `Dst Port`
- `Fwd Pkts/s`
- `Bwd Header Len`
- `Pkt Len Mean`
- `Flow Duration`
- `Fwd Header Len`
- `Flow IAT Min`

## Top-10 Information Gain features

- `Bwd Header Len`
- `Fwd Header Len`
- `Init Bwd Win Byts`
- `Dst Port`
- `Pkt Len Max`
- `Pkt Len Std`
- `Pkt Len Var`
- `Pkt Len Mean`
- `Pkt Size Avg`
- `Bwd Pkt Len Mean`

## Validation metrics

| Feature set | Macro-F1 | Accuracy |
|---|---:|---:|
| All features | 0.956269 | 0.998384 |
| Without Dst Port | 0.945341 | 0.997866 |
| Without Dst Port and Protocol | 0.945329 | 0.997836 |

Features were removed from the stored training and validation matrices only; all other model parameters use scikit-learn defaults.
