# pipeline_output

Contains the trained model artifacts loaded by the API at startup. Do not modify any file here.

## Folder: `LightGBM_06-May-2026_18-04-07`

Named after the model and the timestamp when training completed (06 May 2026, 18:04:07).

| File | Purpose |
|---|---|
| `best_model.joblib` | Trained LightGBM classifier |
| `scaler.joblib` | RobustScaler fit on training data — applied to every inference request |
| `label_encoder.joblib` | Maps integer class index → class name string (e.g. 2 → "DEPRESSION") |
| `encoding_artifacts.joblib` | Categorical encoding maps used during training |
| `outlier_transformers.joblib` | Per-column outlier smoothing params (strategy + fitted transformer per feature) |
| `feature_names.json` | Ordered list of 28 feature names the model uses for prediction |
| `model_metadata.json` | Model hyperparameters, class names, feature count, test metrics |
| `pipeline_state.json` | Full training pipeline log — outlier stats, feature selection steps, CV results |

## Classes predicted

`ANXIETY`, `BIPOLAR`, `DEPRESSION`, `PHOBIA`, `SUICIDAL_TENDENCY`

## Model performance

| Metric | Score |
|---|---|
| Accuracy | 94.7% |
| F1 (macro) | 94.68% |
| CV F1 (macro) | 94.62% |

## Features used (28 of 52 input fields)

The model was trained on a selected subset of 28 features from the full 52-field input schema:

`hapax_legoman_ratio`, `sentence_count`, `average_sentence_length`, `noun_ratio`, `verb_ratio`, `adverb_ratio`, `first_person_singular_pronoun_frequency`, `modal_verb_frequency`, `negative_frequency`, `parse_tree_depth`, `subordinate_clause_ratio`, `overall_sentiment_score`, `fear_word_frequency`, `sadness_word_frequency`, `anger_word_frequency`, `max_negative_emotion`, `negative_emotion_spike_count`, `sentiment_trajectory_slope`, `emotional_volatility_score`, `absolutist_word_frequency`, `catastrophizing_indicators`, `external_locus_of_control_score`, `uncertainty_word_frequency`, `avoidance_language_frequency`, `past_focused_word_ratio`, `present_focused_word_ratio`, `future_focused_word_ratio`, `cognitive_load_score`

