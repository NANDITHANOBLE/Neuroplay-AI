


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

Established benchmark performance before deep learning (Phase 9-11):

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
