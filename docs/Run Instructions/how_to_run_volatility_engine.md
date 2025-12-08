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

# 1. 📦 Prerequisites

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

# 3. ▶️ Step 1 — Generate Global Features

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
