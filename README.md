# 🧠 NeuroPlay-AI

> **Adaptive Human Behavior Prediction and Strategy Intelligence Platform**
>
> An end-to-end AI system that predicts human decision-making patterns in Rock-Paper-Scissors using Machine Learning, Deep Learning, Explainable AI, Reinforcement Learning, Concept Drift Detection, Computer Vision, FastAPI, PostgreSQL, and Streamlit.

---

# 📖 Project Overview

NeuroPlay-AI is an advanced behavioral intelligence platform designed to study, predict, and adapt to human gameplay patterns.

Unlike traditional Rock-Paper-Scissors bots that rely on fixed rules, NeuroPlay-AI learns behavioral tendencies, detects strategy shifts, explains its predictions, and continuously improves through reinforcement learning.

The project combines multiple AI disciplines into a single production-style system:

- Machine Learning
- Deep Learning
- Explainable AI (XAI)
- Reinforcement Learning
- Concept Drift Detection
- Computer Vision
- FastAPI Backend
- PostgreSQL Database
- Streamlit Dashboard

---

# 🎯 Core Objectives

NeuroPlay-AI aims to answer three key questions:

### 1. Can human behavior be predicted?

Predict a player's next move using historical gameplay patterns.

### 2. Can behavioral strategies be explained?

Provide explainable insights into why a prediction was made.

### 3. Can AI adapt when behavior changes?

Detect strategy shifts in real-time and adjust predictions accordingly.

---

# 🏗️ System Architecture

```text
Player
  │
  ▼

Streamlit Frontend / Webcam Interface
  │
  ▼

FastAPI Backend
  │
  ├────────────────────┬────────────────────┐
  ▼                    ▼                    ▼

ANN Predictor      SHAP Engine      Psychology Profiler

  │                    │                    │
  └──────────┬─────────┴──────────┬─────────┘
             ▼                    ▼

      PostgreSQL Database     Drift Detection
                              (ADWIN / DDM)

             │
             ▼

      Reinforcement Learning
           DQN Agent

             │
             ▼

     Adaptive Game Strategy
```

---

# 🚀 Key Features

## 🎮 Intelligent Gameplay Engine

- Rock-Paper-Scissors game environment
- Human vs AI gameplay
- Match history tracking
- Real-time predictions

---

## 🧠 Human Behavior Prediction

Predicts a player's next move using:

- Historical move patterns
- Win/Loss behavior
- Frequency bias
- Sequence complexity
- Rolling performance metrics

---

## 🔍 Explainable AI (XAI)

SHAP explanations provide insight into:

- Why a move was predicted
- Which behavioral patterns influenced the prediction
- Feature-level contribution scores

Example drivers:

- Previous move patterns
- Win-stay/Lose-shift tendencies
- Sequence complexity
- Move frequency bias

---

## 🌊 Concept Drift Detection

Detects behavioral changes during gameplay.

Implemented algorithms:

- ADWIN
- DDM

Production selection:

✅ ADWIN

Performance:

| Detector | Detection Rate | Average Latency |
|-----------|----------------|----------------|
| ADWIN | 95% | 16.1 rounds |
| DDM | 5% | N/A |

---

## 🤖 Reinforcement Learning Agent

A Deep Q-Network (DQN) learns optimal counter-strategies.

Features:

- Adaptive gameplay
- Reward optimization
- Persona exploitation
- Dynamic strategy selection

### Training Results

| Episode | Average Reward |
|----------|---------------|
| 50 | 2.28 |
| 200 | 18.48 |
| 350 | 29.52 |
| 500 | 35.50 |

---

## 📷 Computer Vision Gameplay

Real-time hand gesture detection using:

- MediaPipe Hand Landmarks
- Webcam Input
- Gesture Recognition

Recognized gestures:

- Rock ✊
- Paper ✋
- Scissors ✌️

---

## 📊 Psychology Dashboard

Classifies player behavioral styles.

Detected patterns:

| Pattern | Description |
|-----------|------------|
| Win-Stay / Lose-Shift | Repeats winning moves |
| Frequency Biased | Overuses one move |
| Cyclic | Follows predictable rotation |
| Unpredictable | High randomness |

---

## 📈 Analytics Dashboard

Visual gameplay analytics powered by Plotly.

Includes:

- Win/Loss/Draw Distribution
- Move Frequency Analysis
- Rolling Win Rate Trends
- Performance Metrics

---

# 🗄️ Database Design

NeuroPlay-AI uses:

- SQLite (Development)
- PostgreSQL (Production)

## Core Tables

### users
Player profiles and aggregate statistics.

### matches
Individual game sessions.

### moves
Round-level gameplay history.

### predictions
Stored model outputs and explanations.

### drift_events
Detected concept drift incidents.

### psychology_profiles
Behavioral classification results.

---

# 🎲 Synthetic Dataset Generation

Since publicly available datasets do not exist for human RPS psychology, NeuroPlay-AI generates realistic training data using six behavioral personas:

1. Random
2. Win-Stay / Lose-Shift
3. Cyclic
4. Frequency Biased
5. Markov Order-2
6. Drifting Bot

Generate dataset:

```bash
python -m neuroplay.data_generation.generate_dataset
```

---

# 🔧 Feature Engineering

Raw gameplay sequences are transformed into behavioral indicators.

Engineered features include:

- Win/Loss streak counters
- Rolling win rates
- Move frequency distributions
- Reaction time statistics
- Lempel-Ziv complexity
- Round position normalization

Run:

```bash
python -m neuroplay.features.feature_pipeline
```

---

# 📊 Model Development

## Baseline Models

| Model | Accuracy |
|---------|----------|
| Random | 32.3% |
| Majority Class | 41.4% |
| Markov Order-1 | 50.1% |
| Markov Order-2 | 57.8% |

---

## ANN Model

Production Model ✅

| Model | Accuracy |
|---------|----------|
| ANN | 70.8% |

Key advantages:

- Highest accuracy
- Most stable performance
- Best sample efficiency

Run:

```bash
python -m neuroplay.models.train_ann
```

---

## LSTM Model

| Model | Accuracy |
|---------|----------|
| LSTM | 65.9% |

Run:

```bash
python -m neuroplay.models.train_lstm
```

---

## Transformer Model

| Model | Accuracy |
|---------|----------|
| Hybrid Transformer | 66.7% |

Run:

```bash
python -m neuroplay.models.train_transformer
```

---

# 🏆 Final Model Comparison

| Model | Accuracy |
|---------|----------|
| Random Baseline | 32.3% |
| Majority Baseline | 41.4% |
| Markov Order-1 | 50.1% |
| Markov Order-2 | 57.8% |
| LSTM | 65.9% |
| Transformer | 66.7% |
| ✅ ANN (Production) | 70.8% |

### Key Finding

The ANN consistently outperformed LSTM and Transformer models due to strong domain-informed feature engineering and better sample efficiency on short behavioral sequences.

---

# 📊 Statistical Evaluation

McNemar's Test confirms ANN performance improvement over Markov Order-2:

```text
p < 0.0001
```

This demonstrates that the performance gain is statistically significant and not caused by random chance.

---

# 🚀 FastAPI Backend

The backend exposes model predictions, gameplay services, and analytics through REST APIs.

## Endpoints

| Endpoint | Method | Purpose |
|------------|---------|----------|
| `/health` | GET | Health check |
| `/game/start` | POST | Start match |
| `/game/play` | POST | Submit move |
| `/game/{match_id}/history` | GET | Match history |
| `/leaderboard` | GET | Rankings |
| `/explain/{move_id}` | GET | SHAP explanation |

Start API:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# 🎨 Streamlit Frontend

The frontend communicates with FastAPI entirely through HTTP.

### Pages

- 🎮 Play
- 📷 Webcam Mode
- 🏆 Leaderboard
- 📊 Analytics
- 🧬 Psychology Dashboard

Run:

```bash
streamlit run frontend/streamlit_app/app.py
```

---

# 🧪 Testing

Comprehensive automated testing across:

- Feature Engineering
- Baseline Models
- Synthetic Personas
- Database Layer
- FastAPI Endpoints
- Drift Detection
- Data Preprocessing
- Evaluation Pipeline

Run tests:

```bash
python -m pytest --cov=src --cov=backend --cov-report=term-missing
```

### Current Coverage

```text
52%
```

---

# 📂 Project Structure

```text
NeuroPlay-AI/
│
├── backend/
├── frontend/
├── neuroplay/
│   ├── data_generation/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── explainability/
│   ├── drift/
│   ├── rl/
│   ├── cv/
│   ├── db/
│   └── evaluation/
│
├── database/
├── docs/
├── notebooks/
├── tests/
├── docker/
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

# 🐳 Deployment

Dockerized deployment support included.

Services:

- FastAPI Backend
- Streamlit Frontend
- PostgreSQL Database

Run:

```bash
docker compose up --build
```

---

# 🌟 Project Highlights

✅ Human Behavior Prediction

✅ Explainable AI (SHAP)

✅ Concept Drift Detection

✅ Reinforcement Learning

✅ Computer Vision Integration

✅ FastAPI Backend

✅ Streamlit Dashboard

✅ PostgreSQL Database

✅ Statistical Model Evaluation

✅ Docker Deployment

✅ End-to-End AI System

---

# 📜 License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
