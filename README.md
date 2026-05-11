# Mindspace — Text Sentiment API

A FastAPI inference server that predicts a mental health profile from 52 pre-extracted text/speech features using a trained LightGBM model.

## What it does

Accepts 52 numerical features extracted from a speech or text sample and returns the most likely mental health profile out of 5 classes: `ANXIETY`, `BIPOLAR`, `DEPRESSION`, `PHOBIA`, `SUICIDAL_TENDENCY`.

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | Yes | Service info — name, supported classes, feature count |
| GET | `/health` | No | Health check — returns `{"status": "ok"}` when ready |
| POST | `/predict` | Yes | Run prediction on 52 input features |
| GET | `/model/info` | Yes | Model structure info (feature names, classes, scaler) |

## Input features

The `/predict` endpoint expects 52 float fields grouped as:

- **Lexical / surface (6):** `total_word_count`, `unique_word_count`, `type_token_ratio_ttr`, `moving_average_ttr`, `hapax_legoman_ratio`, `repetition_rate`
- **Syntactic (9):** `sentence_count`, `average_sentence_length`, `noun_ratio`, `verb_ratio`, `adjective_ratio`, `adverb_ratio`, `first_person_singular_pronoun_frequency`, `modal_verb_frequency`, `negative_frequency`
- **Deep syntactic (4):** `parse_tree_depth`, `avg_dependency_length`, `clause_count`, `subordinate_clause_ratio`
- **Sentiment / emotion (14):** `positive_emotion_word_ratio`, `negative_emotion_word_ratio`, `overall_sentiment_score`, `fear_word_frequency`, `sadness_word_frequency`, `anger_word_frequency`, `joy_frequency`, `disgust_frequency`, `surprise_frequency`, `emotional_intensity_ratio`, `max_negative_emotion`, `negative_emotion_spike_count`, `sentiment_variance`, `sentiment_trajectory_slope`
- **Emotional dynamics (1):** `emotional_volatility_score`
- **Coherence / discourse (4):** `semantic_coherence_score`, `topic_shift_frequency`, `max_sentence_similarity`, `first_last_sentence_similarity`
- **Psychological / cognitive (8):** `absolutist_word_frequency`, `helplessness_phrase_frequency`, `catastrophizing_indicators`, `external_locus_of_control_score`, `rumination_phrase_frequency`, `uncertainty_word_frequency`, `avoidance_language_frequency`, `threat_anticipation_language`
- **Temporal focus (4):** `self_reference_density`, `past_focused_word_ratio`, `present_focused_word_ratio`, `future_focused_word_ratio`
- **Miscellaneous (2):** `filler_word_frequency`, `cognitive_load_score`

See `payload.json` for a complete sample input.

## How to use

### Prerequisites

```
pip install requests python-dotenv
```

### Call with Python `requests`

```python
import requests

url = "http://localhost:9000/predict"

payload = {
    "total_word_count": 756.0,
    "unique_word_count": 361.0,
    "type_token_ratio_ttr": 0.4775132275132275,
    "moving_average_ttr": 0.6340712759117781,
    "hapax_legoman_ratio": 0.2967731849304067,
    "sentence_count": 52.0,
    "average_sentence_length": 2.3686352126669856,
    "repetition_rate": 0.6902180055274213,
    "noun_ratio": 0.119075921429951,
    "verb_ratio": 0.1053030456986042,
    "adjective_ratio": 0.0696799619003486,
    "adverb_ratio": 0.0369357405664167,
    "first_person_singular_pronoun_frequency": 0.0360912175890274,
    "modal_verb_frequency": 0.0105439882897891,
    "negative_frequency": 0.0302877317434468,
    "parse_tree_depth": 3.139033393984063,
    "avg_dependency_length": 2.4262333878835185,
    "clause_count": 18.0,
    "subordinate_clause_ratio": 0.4092288771764105,
    "positive_emotion_word_ratio": 0.4519198303063868,
    "negative_emotion_word_ratio": 0.4747744009542999,
    "overall_sentiment_score": -0.440641413695465,
    "fear_word_frequency": 0.8018379677415697,
    "sadness_word_frequency": 0.1156283330711261,
    "anger_word_frequency": 0.2595347173804833,
    "joy_frequency": 0.8998598145993033,
    "disgust_frequency": 0.025069735268611,
    "surprise_frequency": 0.1004618858545713,
    "emotional_intensity_ratio": 1.0,
    "max_negative_emotion": 0.8018379677415697,
    "negative_emotion_spike_count": 4.0,
    "sentiment_variance": 0.1059463201659939,
    "sentiment_trajectory_slope": -0.0027067640342667,
    "emotional_volatility_score": 0.7623467828488708,
    "semantic_coherence_score": 0.020197040586832,
    "topic_shift_frequency": 0.3997036421355432,
    "max_sentence_similarity": 0.8644514023598229,
    "first_last_sentence_similarity": 0.0417282281960346,
    "absolutist_word_frequency": 0.1692641149848399,
    "helplessness_phrase_frequency": 0.3847258186272604,
    "catastrophizing_indicators": 0.4902486769242626,
    "external_locus_of_control_score": 0.0,
    "rumination_phrase_frequency": 0.3520795139528316,
    "uncertainty_word_frequency": 0.6252863650474074,
    "avoidance_language_frequency": 0.4899296465048904,
    "threat_anticipation_language": 0.4676165927461236,
    "self_reference_density": 0.0816041396076135,
    "past_focused_word_ratio": 0.0,
    "present_focused_word_ratio": 0.4327344615168387,
    "future_focused_word_ratio": 0.4004957450050325,
    "filler_word_frequency": 0.0581001570433341,
    "cognitive_load_score": 31.3483649508134
}

headers = {
    "X-API-Key": "your_api_key"
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
```

### Ready-to-run script

```bash
python test_predict_api.py
```

`test_predict_api.py` reads `payload.json`, calls all 4 endpoints, prints results to the console, and saves a timestamped JSON file to `test_api_output/`. Requires `.env` with `MINDSPACE_TEXT_API_KEY` set.

## Setup

### 1. Configure environment

```bash
cp example.env .env
# Edit .env and set MINDSPACE_TEXT_API_KEY to your key
```

### 2. Run with Docker (recommended)

```bash
docker compose up --build
```

### 3. Run locally

```bash
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 9000
```

Interactive docs available at `http://localhost:9000/docs`.

## File structure

```
deployment-text/
├── main.py                        # FastAPI application
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker image (Gunicorn + Uvicorn workers)
├── docker-compose.yml             # Single-service compose file
├── payload.json                   # Sample input for /predict (28 model features)
├── test_predict_api.py            # Ready-to-run test script
├── example.env                    # Environment variable template
├── .env                           # Your actual keys (never commit this)
├── test_api_output/               # Timestamped test run outputs (auto-created)
├── demo-api-input-data-sample/    # 2 sample inputs per class (10 files total)
│   ├── anxiety_sample_1.json
│   ├── anxiety_sample_2.json
│   ├── bipolar_mania_sample_1.json
│   ├── bipolar_mania_sample_2.json
│   ├── depression_sample_1.json
│   ├── depression_sample_2.json
│   ├── phobia_sample_1.json
│   ├── phobia_sample_2.json
│   ├── suicidal_tendency_sample_1.json
│   └── suicidal_tendency_sample_2.json
└── pipeline_output/               # Trained model artifacts
    └── LightGBM_06-May-2026_18-04-07/
```

## Model performance

| Metric | Score |
|---|---|
| Accuracy | 94.7% |
| F1 (macro) | 94.68% |
| F1 (weighted) | 94.68% |
| CV F1 (macro) | 94.62% |

Model: LightGBM — trained and selected via 5-fold cross-validation. Scaler: RobustScaler.

