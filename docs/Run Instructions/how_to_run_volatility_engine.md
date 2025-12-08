# 🏃‍♂️ How to Run the Volatility Forecasting Engine

**Execution Guide**

This document explains how to run each component of the volatility pipeline in the correct order:
1. Global feature generation
2. Global feature validation
3. Global feature pruning
4. Per-ticker validation
5. Multi-ticker model training & forecasting
6. Event enrichment
7. Signal generation

Designed so anyone can execute the full workflow without guessing.

---

# 📦 Prerequisites

### ✔ Install Python 3.9–3.11  
Any version in this range will work.

### ✔ Install required Python packages  
From the repository root:

```bash
pip install -r requirements.txt
```

### ✔ Core dependencies used in the model
- numpy  
- pandas  
- xgboost  
- scikit-learn
- tensorflow
- ta (indicators)

---
```
volatility-engine/
│
├── src/
│   ├── 01_Global_Feature_Generator.py
│   ├── 02_Global_Feature_Validator.py
│   ├── 03_Global_Feature_Pruner.py
│   ├── 04_Global_Ticker_Validation_Scanner.py
│   ├── 05_Multi_Ticker_Model_Runner.py
│   ├── 06_Event_Engine.py
│   ├── 07_Signal_Engine.py
│   └── utils/
│
├── batch_vol_runs/
│   ├── global_features/
│   ├── feature_pruner/
│   ├── per_ticker_forecasts/
│   ├── summary_metrics.csv
│   ├── summary_with_events.csv
│   ├── trade_signals_today.csv
│   └── events_manual.csv
│
└── docs/
    └── how_to_run_volatility.md  ← You are here
```
---

# ▶️ Step 1 — Generate Global Features

This script takes raw input price/IV data and creates the full feature set:
- Rolling volatility (1–252 days)
- Ratios, spreads, entropy
- Macro indices (VIX, VXN, MOVE)
- Range/ATR/Bandwidth
- Event placeholders

Run:

```
python src/01_Global_Feature_Generator.py
```

Outputs:
```
batch_vol_runs/global_features/all_tickers_raw.parquet
```

This is the master dataset used everywhere.

---

# 🔍 Step 2 — Validate Global Features

This script checks for structural issues:
- Missing target values (rv_14_forward)
- Empty symbols
- NaNs or invalid values
- Duplicate rows
- Date ordering
- Deterministic feature counts

Run:

```
python src/02_Global_Feature_Validator.py
```

Outputs:

```
batch_vol_runs/global_features/validation_report.txt
batch_vol_runs/global_features/validation_flags.csv
```

If validation fails → fix the source data before continuing.

---

# ✂️ Step 3 — Global Feature Pruner

Prunes features once globally, across all tickers:

-Drop high-missing columns
-Drop zero-variance columns
-Correlation cluster pruning
-XGBoost gain scoring
-Linear coefficients + correlations
-Economic sign alignment
-Combined Feature Quality Score (0–100)

Run:

```
python src/03_Global_Feature_Pruner.py
```

Outputs:

```
batch_vol_runs/feature_pruner/
    pruner_missingness_report.csv
    pruner_corr_matrix.csv
    feature_scores.csv
    selected_features.csv
    selected_feature_list.txt
```

selected_feature_list.txt is the REQUIRED feature set for the model runner.

---

# 🔎 Step 4 — Global Ticker Validation Scanner

This checks which tickers are fit for modelling using:

- Target rows available (for WFA)
- Feature missingness per ticker
- LSTM sequence viability
- Latest row completeness
- Covariance stability (Ledoit–Wolf)
- IV/RV availability

Run:

```
python src/04_Global_Ticker_Validation_Scanner.py
```

Outputs:

```
batch_vol_runs/ticker_validation_report.csv
batch_vol_runs/valid_tickers_for_training.csv
```

Only tickers in valid_tickers_for_training.csv proceed to modelling.

---

# 🚀 Step 5 — Multi-Ticker Model Runner

This is the core forecasting engine:

- Trains global XGB quantile models
- Trains time-decay LSTM
- Performs walk-forward evaluation
- Computes regime confidence
- Computes Mahalanobis drift
- Combines models into an ensemble
- Exports dashboard CSVs

Run:

```
python src/05_Multi_Ticker_Model_Runner.py
```

Outputs:

```
batch_vol_runs/summary_metrics.csv
batch_vol_runs/per_ticker_forecasts/<SYMBOL>_dashboard_data.csv
```

summary_metrics.csv is the input for the event engine.

---

# 📅 Step 6 — Event Engine

This script enriches forecasts with event information.

It consumes:

✔ summary_metrics.csv
✔ events_manual.csv

Adds:

- Next event date
- Type (CPI, FOMC, OPEX, VIX expiry, etc.)
- Importance
- Days until event
- Event window active (yes/no)
- Pressure scores

Run:

```
python src/06_Event_Engine.py
```

Outputs:

```
batch_vol_runs/summary_with_events.csv
```

# 🔧 Maintaining the Events Manual Calendar

Open:

```
batch_vol_runs/events_manual.csv
```

| Column      | Meaning                        |
| ----------- | ------------------------------ |
| event_date  | dd/mm/yyyy                     |
| symbol      | Ticker or ALL                  |
| scope       | GLOBAL or SYMBOL               |
| event_type  | CPI, PPI, NFP, FOMC, OPEX, etc |
| importance  | 1–5                            |
| window_pre  | days before                    |
| window_post | days after                     |
| applies_to  | ALL or symbol                  |
| source      | optional reference             |

---

# 📈 Step 7 — Signal Engine

This produces daily trade opportunities:

- Long Gamma entries
- Short Vega entries
- Exit conditions
- Regime tagging
- Diversification tags
- Sorting by IV mispricing

Run:

```
python src/07_Signal_Engine.py
```

Outputs:

```
batch_vol_runs/trade_signals_today.csv
```

This is a execution-ready table.

# 🎉 End-to-End Pipeline Complete

You now have:

✔ Valid features
✔ Valid tickers
✔ Trained models
✔ Ensemble forecasts
✔ Events applied
✔ Trade signals generated
✔ Dashboard-ready output
