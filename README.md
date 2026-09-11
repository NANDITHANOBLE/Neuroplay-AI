


## 🗄️ Database Schema

NeuroPlay-AI uses a relational schema (SQLite for dev, PostgreSQL for production):

- **users** — player profiles, aggregate stats
- **matches** — one row per game session
- **moves** — every individual round (the core time-series data)
- **predictions** — logged model outputs per round, with explainability data
- **drift_events** — concept drift detections (ADWIN/DDM)
- **psychology_profiles** — behavioral classification per user

See `docs/ARCHITECTURE.md` for the full ER diagram.

## 🎲 Synthetic Dataset Generation

Since no public dataset exists for human RPS psychology, NeuroPlay-AI bootstraps
model training using 6 research-grounded synthetic personas:
Random, Win-Stay/Lose-Shift, Cyclic, Frequency-Biased, Markov-Order-2, and
Drifting (mid-match strategy switch, used to validate concept drift detection).

Run: `python -m neuroplay.data_generation.generate_dataset`

## 🧹 Data Preprocessing

Raw gameplay data is loaded from the database, transformed into sliding-window
sequences (default window size = 5 rounds), and split into train/val/test sets
at the match level (stratified by persona) to prevent temporal data leakage.

Run: `python -m neuroplay.preprocessing.preprocess_pipeline`


## 📊 Exploratory Data Analysis

Validates synthetic persona behavioral signatures via move distribution,
win/loss/draw rates, move-transition heatmaps, and Lempel-Ziv sequence
complexity — confirming each of the 6 personas (Random, Win-Stay/Lose-Shift,
Cyclic, Frequency-Biased, Markov-Order-2, Drifting) exhibits its intended
distinct behavioral pattern before proceeding to modeling.

Run: `jupyter notebook notebooks/01_exploratory_data_analysis.ipynb`


## 🔧 Feature Engineering

Raw move sequences are transformed into behavioral features:
- Signed win/loss streak counters (win-stay/lose-shift signal)
- Rolling win-rate (5 & 20 round windows)
- Rolling move frequency distribution
- Reaction time statistics (mean, std)
- Rolling Lempel-Ziv complexity (predictability signal)
- Normalized round position

Run: `python -m neuroplay.features.feature_pipeline`


## 🎯 Baseline Models

Established benchmark performance before deep learning :

| Model | Accuracy | Win Rate |
|---|---|---|
| Random | 32.3% | 32.3% |
| Majority Class | 41.4% | 41.4% |
| Markov Order-1 | 50.1% | 50.1% |
| **Markov Order-2** | **57.8%** | **57.8%** |

Markov-2 nearly doubles the random baseline, confirming genuine exploitable
behavioral patterns exist in the synthetic persona data — validating the
entire pipeline before investing in deep learning.

Run: `python -m neuroplay.models.run_baselines`


## 🧠 ANN Model

A feed-forward neural network (PyTorch) trained on  engineered
behavioral features, using z-score normalization (critical fix — without
it, the model collapsed to constant-output prediction).

| Model | Accuracy |
|---|---|
| Random Baseline | 32.3% |
| Majority Baseline | 41.4% |
| Markov Order-1 | 50.1% |
| Markov Order-2 | 57.8% |
| **ANN ** | **70.8%** |

Run: `python -m neuroplay.models.train_ann`


## 🔮 LSTM Model

An LSTM consuming raw move sequences (5-round window) with learned embeddings
for player/AI moves.

| Model | Test Accuracy |
|---|---|
| Random Baseline | 32.3% |
| Majority Baseline | 41.4% |
| Markov Order-1 | 50.1% |
| Markov Order-2 | 57.8% |
| LSTM (Phase 10) | 65.9% |
| **ANN ** | **70.8%** |

**Key finding:** The ANN outperformed the LSTM despite the LSTM having access to
raw sequential order. This suggests that on this dataset size (~8K training rows)
and window length (5 rounds), explicit feature engineering (streaks, rolling
win-rate, LZ complexity) is more sample-efficient than requiring the LSTM to
discover these temporal patterns independently. The LSTM still meaningfully
beats the classical Markov-2 baseline, confirming genuine sequence-modeling value.

Run: `python -m neuroplay.models.train_lstm`


## 🎯 Model Comparison — Final Results

| Model | Test Accuracy | Input |
|---|---|---|
| Random Baseline | 32.3% | None |
| Majority Baseline | 41.4% | None |
| Markov Order-1 | 50.1% | Last 1 move |
| Markov Order-2 | 57.8% | Last 2 moves |
| LSTM | 65.9% | Raw 5-move sequence |
| Hybrid Transformer | 66.7% | Sequence + engineered features |
| **ANN (Production Model)** | **70.8%** | Engineered features |

**Key finding:** For short-horizon (5-round) behavioral sequence prediction on
this dataset, explicit feature engineering (streaks, rolling win-rate, LZ
complexity) consistently outperforms both LSTM and Transformer sequence models
— even when the Transformer has access to the same engineered features. This
demonstrates that architectural sophistication alone doesn't guarantee better
performance; sample efficiency and domain-informed feature design matter more
at this data scale. **The ANN is selected as the production model** going into
Evaluation Framework and beyond.

Run: `python -m neuroplay.models.train_transformer`


## 📊 Evaluation Framework

Formalized model comparison with statistical rigor:

| Model | Accuracy |
|---|---|
| Random Baseline | 35.9% |
| Majority Baseline | 41.4% |
| Markov Order-1 | 50.1% |
| Markov Order-2 | 57.8% |
| **ANN (Selected)** | **70.8%** |

**McNemar's test** confirms the ANN's improvement over Markov Order-2 is
statistically significant (p < 0.0001) — not due to random chance.

*Note: LSTM (65.9%) and Transformer (66.7%) are tracked
separately and excluded from this automated leaderboard; both underperformed
the ANN and are not candidates for production selection.*

The ANN is saved as `models/selected_model.json` — the canonical reference
used by Explainable AI,RL Agent and
FastAPI Backend.

Run: `python -m neuroplay.evaluation.run_full_evaluation`

## 🔍 Explainable AI

SHAP (GradientExplainer) provides per-prediction feature attribution for the
production ANN model — answering "why did the model predict this move?"

Example: for a predicted Scissors move, the model relied most heavily on:
- Previous round's move (`last_move_is_paper`, `last_move_is_rock`)
- Rolling sequence complexity (`rolling_lz_complexity_10`)
- Recent move frequency bias (`move_freq_scissors`)

This confirms the model's reasoning aligns with the behavioral patterns
(win-stay/lose-shift, frequency bias) validated in  EDA.

Run: `python -m neuroplay.explainability.run_explainability`


## 🌊 Concept Drift Detection

Two detectors implemented and validated against  `DriftingBot`
ground truth (persona switch at round 50):

| Detector | Detection Rate | Avg Latency |
|---|---|---|
| **ADWIN (selected)** | **95%** | **16.1 rounds** |
| DDM | 5% | N/A (too conservative) |

**Finding:** DDM's 3-sigma threshold is calibrated for low-baseline-error
classifiers. Our ANN's ~29% baseline error rate (70.8% accuracy) makes the
persona-switch signal too subtle relative to DDM's strict threshold. ADWIN's
sub-window comparison approach proved more robust to this noisier stream and
is selected as the production drift detector.

Run: `python -m neuroplay.drift.run_drift_detection`

## 🤖 Reinforcement Learning Agent

A DQN agent learns optimal counter-strategy policy (not just "counter the
prediction") by training against a mixed pool of persona bots, with reward
+1/win, 0/draw, -1/loss.

**Training progress (500 episodes):**

| Episode | Avg Reward (last 50) | Epsilon |
|---|---|---|
| 50 | 2.28 | 0.778 |
| 200 | 18.48 | 0.367 |
| 350 | 29.52 | 0.173 |
| 500 | **35.50** | 0.082 |

Clean, monotonic improvement as exploration (epsilon) decays — confirming the
agent successfully learns to exploit exploitable personas (Cyclic, Markov-2,
Frequency-Biased) rather than playing a static counter-strategy.

Run: `python -m neuroplay.rl.train_dqn`

## 📷 Computer Vision Module

Rule-based Rock/Paper/Scissors gesture classification using MediaPipe hand
landmarks (21-point detection) + a finger-extension heuristic — no custom
model training required.

**Debugging note:** Initial Scissors detection failed due to an overly
strict requirement that the ring finger be fully curled — biomechanically
inconsistent across users and sensitive to hand rotation. Relaxed to require
only index+middle clearly extended and pinky curled, tolerating ring-finger
ambiguity.

⚠️ Also required pinning `mediapipe==0.10.14` — the latest release (1.0.1)
replaced the legacy `solutions.hands` API with a new Tasks API, breaking
this implementation.

Run: `python -m neuroplay.cv.run_webcam_demo`

## 🚀 FastAPI Backend

REST API unifying the ANN predictor, database, and game logic behind clean
endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness check |
| `/game/start` | POST | Start a new match |
| `/game/play` | POST | Submit a move, get AI response |
| `/game/{match_id}/history` | GET | Full match history |
| `/leaderboard` | GET | Top users by win rate |
| `/explain/{move_id}` | GET | SHAP explanation for a prediction |

Models load once at startup (via `lru_cache` dependency injection) — not
per-request — critical for API latency.

**Debugging note:** Discovered `isort` and `ruff` maintain *separate*
independent configs for import-sorting behavior. An aliased combined import
(`from x import A, B as C`) caused an infinite fix-loop until
`combine-as-imports = true` was set in **both** `[tool.isort]` and
`[tool.ruff.lint.isort]` in `pyproject.toml`.

Run: `uvicorn backend.app.main:app --reload --port 8000`
Interactive docs: `http://127.0.0.1:8000/docs`
