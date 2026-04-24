# pipeline_output

Contains the trained model artifacts loaded by the API at startup. Do not modify any file here.

## Folder: `LightGBM_13032026_110356`

Named after the model and the timestamp when training completed (13 March 2026, 11:03:56).

| File | Purpose |
|---|---|
| `best_model.joblib` | Trained LightGBM classifier |
| `scaler.joblib` | RobustScaler fit on 40,000 training rows — applied to every inference request |
| `label_encoder.joblib` | Maps integer class index → class name string (e.g. 2 → "Depression") |
| `encoding_artifacts.joblib` | Categorical encoding maps used during training |
| `outlier_transformers.joblib` | Per-column outlier smoothing params (strategy + fitted transformer per feature) |
| `feature_names.json` | Ordered list of 43 feature names the model expects |
| `model_metadata.json` | Model hyperparameters, class names, feature count |
| `pipeline_state.json` | Full training pipeline log — outlier stats, feature selection steps, CV results |

## Classes predicted

`Anxiety`, `Bipolar_Mania`, `Depression`, `Normal`, `Phobia`, `Stress`, `Suicidal_Tendency`
