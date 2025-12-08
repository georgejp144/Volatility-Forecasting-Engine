## 📈 Volatility Forecasting Engine

**A reproducible, institution-style forecasting and signal pipeline for 14-day realised volatility.**

This repository contains the full implementation of the engine, including:

1. Cross-asset data ingestion & validation
2. Feature engineering, pruning & economic sign-checks
3. Hybrid modelling architecture (XGBoost + LSTM + Regime-Weighted Ensemble)
4. Event-aware signal generation (long gamma / short vega)
5. Daily system outputs & dashboards

The engine provides:

1. Forward RV₁₄ forecasts
2. P10 / P50 / P90 uncertainty bands
3. IV–RV mispricing detection
4. Regime classification & Model Reliability
5. Entry/exit conditions for gamma/vega strategies

---

## 🚀 Key Features

- Deterministic, audit-ready daily forecasting
- Hybrid architecture: XGBoost + LSTM + ensemble weighting 
- Detailed Proposal - Volatility …
- Event-awareness: CPI, NFP, FOMC, OPEX, VIX expiry
- Institutional-grade data validation
- Signal gating & risk controls
- Execution-ready CSV outputs

The forecasting engine supports data-driven long gamma (RV > IV) and short vega (IV > RV) decisions.

---

## 🧱 High-Level Architecture

The pipeline follows a seven-layer design

1. Data Ingestion
OHLC, RV windows, IV surfaces, macro/cross-asset series
2. Validation Layer
Calendar integrity, missingness, anomalies, cross-ticker consistency
3. Feature Pruning
Missingness, variance, correlation clustering, XGB gain, economic-sign checks
4. Universe Eligibility
Covariance stability (Ledoit-Wolf), sequence viability, latest row completeness
5. Forecasting Engine
- Cross-sectional XGBoost
- Sequential LSTM
- Regime-aware ensemble weighting
6. Event & Regime Layer
Macro windows, vol-structure events, proximity scoring, Mahalanobis drift
7. Signal & Execution Layer
Long gamma / short vega logic, exit rules, diversification tags

---

## 📁 Folder Structure Overview

Your repository is organised as:

```
volatility-forecasting-engine/
│
├── src/
│   ├── 01_synthetic_game_generator.py
│   ├── 02_feature_validator.py
│   ├── 03_game_feature_pruner.py
│   └── 04_model_runner.py
│   
│
├── data/
│   ├── synthetic_examples/
│   │   ├── 
│   │   ├── 
│   │   ├── 
│   │   └── 
│   └── (full datasets if available)
│
└── docs/
    ├── how_to_run.md
    └── Detailed Proposal - Volatility Forecasting Engine.pdf

```

---   

## 🔮 Model Architecture

The predictive engine is built from three components:

1. XGBoost (Cross-Sectional)
- Learns nonlinear structure across features
- Provides P10 / P50 / P90 forecasts
- Robust to correlated financial variables

2. LSTM (Sequential)
- Learns volatility clustering, regime persistence
- Sliding windows of 22 days
- Early stopping, fixed seeds, dropout regularisation

3. Regime-Weighted Ensemble
- Not a simple average
- Weights driven by:
    -Model Reliability Score (MRS)
    -Regime similarity
    -Walk-forward confidence
    -Drift (Mahalanobis)
    -Uncertainty spreads

Output:

Ensemble_Forecast_RV₁₄, P10/P50/P90, uncertainty bands

---  

## ⚡ Signal Generation

Signals are generated when IV–RV mispricing, forecast confidence and market regime align.

Long Gamma Entry

Enter when:
- IV is below forecast
- Narrow uncertainty band
- RV expected to rise
- Compression environment
- High reliability
- No event window active

Short Vega Entry

Enter when:
- IV is above forecast upper bound
- Low uncertainty
- High reliability
- No macro event active

Exit Logic

Exit long gamma when any of:
- Expected RV < Current RV
- IV rises above forecast
- Uncertainty widens
- Macro event within 3 days

---  

## 📊 System Outputs

Daily files include:
- Forecast_RV₁₄ (LSTM & XGB & Ensemble)
- P10 / P50 / P90
- Ensemble_spread & Ensemble_vs_IV
- Model Reliability Score
- Regime & Event flags
- Trade signals (LongGamma, ShortVega)

Each ticker generates a CSV:

vol_dashboard_data_<SYMBOL>.csv

Plus a universe-level snapshot:

summary_metrics.csv

---  

## 🛡 Risk Controls & Reliability

The engine includes:
- Walk-forward validation
- Regime similarity checks
- Drift scores
- Event blackout windows
- Uncertainty thresholds
- Reliability score gating

Signals only occur when:
- Mispricing exists
- Model has conviction
- Market structure supports hedging
- No major event risk

---  

## ⚠️ Limitations

The system is conservative:
- Volatility remains partially unpredictable
- Event shocks & jumps cannot be forecast perfectly
- Execution risk & transaction costs may reduce edge

👤 Author

## 🛠 Technologies Used

- Python 3.10+
- NumPy, Pandas, SciKit-Learn, XGBoost, TensorFlow
- Walk-forward validation & event engines

---  

👤 Author

George Pearson
