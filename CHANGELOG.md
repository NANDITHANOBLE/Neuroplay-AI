# Changelog

## [1.0.0] - 2026-09-15
### Added
- Complete  build: synthetic data generation, EDA, feature engineering,
  4 model architectures (Markov, ANN, LSTM, Transformer), evaluation framework,
  SHAP explainability, ADWIN/DDM concept drift detection, DQN RL agent,
  MediaPipe computer vision, FastAPI backend, Streamlit frontend with 5 pages,
  Alembic database migrations, Docker deployment configs, 28-test suite.

### Key Results
- Production ANN model: 70.8% test accuracy (statistically significant vs.
  57.8% Markov-2 baseline, McNemar's p < 0.0001)
- ADWIN drift detection: 95% detection rate, 16.1 round average latency
- DQN agent: average reward improved 2.28 → 35.50 over 500 episodes

### Known Limitations
- Docker configs untested in local dev environment (no Docker available)
- Test coverage 52% (focused on business logic; training scripts and
  visualization utilities excluded by design)
