# ============================================================
# Volatility Forecasting Engine — Global Feature Pruner
# ============================================================
#
# PURPOSE:
#   Dimensionality reduction & feature scoring for the global dataset:
#     • Numeric-only cleaning
#     • Missingness & zero-variance filters
#     • Correlation cluster pruning
#     • XGBoost gain scoring
#     • Linear explainability (standardised coefficients + correlations)
#     • Economic sign alignment
#     • Combined Feature Quality Score (0–100)
#     • Final feature selection (Top-N + Score threshold)
#
# INPUT:
#   batch_vol_runs/global_features/all_tickers_raw.parquet
#
# OUTPUTS:
#   batch_vol_runs/feature_pruner/
#       pruner_missingness_report.csv
#       pruner_corr_matrix.csv
#       feature_scores.csv
#       selected_features.csv
#       selected_feature_list.txt
#
# NOTES:
#   • GLOBAL PRUNING (all tickers pooled)
#   • This is NOT the forecasting model, only feature reduction
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# =============================================================
# CONFIG
# =============================================================
RAW_PATH = "batch_vol_runs/global_features/all_tickers_raw.parquet"
OUT_DIR = "batch_vol_runs/feature_pruner"

TARGET = "rv_14_forward"       # name of the target column in all_tickers_raw
MAX_MISSING_PCT = 40.0         # drop features with > this % missing
HIGH_CORR_THRESHOLD = 0.92     # correlation cluster pruning threshold
TOP_N_FEATURES = 20            # base number of features to keep
MIN_SCORE_THRESHOLD = 45.0     # keep any feature with FQS >= this

os.makedirs(OUT_DIR, exist_ok=True)

# Tuned XGBoost profile for pruning (not the same as forecasting model)
PRUNER_XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=4,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=42,
    tree_method="hist",
    n_jobs=1,
)


# =============================================================
# LOAD DATA
# =============================================================
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Could not find raw feature file at: {RAW_PATH}")

df_raw = pd.read_parquet(RAW_PATH)
print(f"[PRUNER] Loaded raw dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

if TARGET not in df_raw.columns:
    raise ValueError(f"[PRUNER] Target column '{TARGET}' not found in dataset")

# Drop rows with missing target
df_raw = df_raw.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"[PRUNER] After dropping NaN target rows: {df_raw.shape[0]} rows")

# =============================================================
# NUMERIC-ONLY CLEANING (FIX FOR STRING COLUMNS LIKE 'symbol')
# =============================================================
# Save target separately
target_series = df_raw[TARGET].copy()

# Keep only numeric columns (drops 'symbol', 'date', etc.)
df_num = df_raw.select_dtypes(include=[np.number]).copy()

# Reattach target if it was dropped by the numeric filter
if TARGET not in df_num.columns:
    df_num[TARGET] = target_series.values

df = df_num.copy()
del df_num

print(f"[PRUNER] After numeric-only filter: {df.shape[0]} rows, {df.shape[1]} numeric columns")

# Sanity check
if df[TARGET].isna().any():
    # If any NaNs reappeared in target, drop them
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"[PRUNER] After cleaning NaN target rows: {df.shape[0]} rows")

# =============================================================
# STEP 1 — MISSINGNESS + ZERO-VARIANCE FILTERS
# =============================================================
feature_cols = [c for c in df.columns if c != TARGET]

missing_pct = df[feature_cols].isna().mean() * 100.0
nunique = df[feature_cols].nunique()

zero_var_features = nunique[nunique <= 1].index.tolist()
too_missing_features = missing_pct[missing_pct > MAX_MISSING_PCT].index.tolist()

keep_features = [
    c
    for c in feature_cols
    if (c not in zero_var_features) and (c not in too_missing_features)
]

print(f"[PRUNER] Features before filters       : {len(feature_cols)}")
print(f"[PRUNER] Zero-variance features dropped: {len(zero_var_features)}")
print(f"[PRUNER] High-missing features dropped : {len(too_missing_features)}")
print(f"[PRUNER] Features after basic filters  : {len(keep_features)}")

# Missingness / variance report
missing_report = pd.DataFrame(
    {
        "Feature": feature_cols,
        "Missing_Pct": missing_pct.reindex(feature_cols).values,
        "N_Unique": nunique.reindex(feature_cols).values,
        "Zero_Var_Flag": [1 if f in zero_var_features else 0 for f in feature_cols],
        "Too_Missing_Flag": [1 if f in too_missing_features else 0 for f in feature_cols],
    }
)
missing_report.to_csv(os.path.join(OUT_DIR, "pruner_missingness_report.csv"), index=False)
print("[PRUNER] Saved pruner_missingness_report.csv")

# Apply filters
df = df[keep_features + [TARGET]]

if len(keep_features) == 0:
    raise RuntimeError("[PRUNER] No features left after missingness/variance filtering")

# =============================================================
# STEP 2 — HIGH-CORRELATION CLUSTER PRUNING
# =============================================================
corr = df[keep_features].corr().abs()

# Save full correlation matrix for inspection
corr.to_csv(os.path.join(OUT_DIR, "pruner_corr_matrix.csv"))
print("[PRUNER] Saved pruner_corr_matrix.csv")

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr_to_drop = [
    column for column in upper.columns if any(upper[column] > HIGH_CORR_THRESHOLD)
]

cluster_pruned_features = [f for f in keep_features if f not in high_corr_to_drop]

print(f"[PRUNER] High-corr features dropped    : {len(high_corr_to_drop)}")
print(f"[PRUNER] Features after corr pruning   : {len(cluster_pruned_features)}")

if len(cluster_pruned_features) < 3:
    # Safety: do not over-prune; fallback to pre-corr version
    print("[PRUNER] Warning: too few features after corr pruning, reverting to pre-corr set")
    cluster_pruned_features = keep_features.copy()

df = df[cluster_pruned_features + [TARGET]]
features = cluster_pruned_features

# =============================================================
# STEP 3 — XGBOOST FEATURE IMPORTANCE (GAIN)
# =============================================================
X = df[features].copy()
y = df[TARGET].values

# Median imputation for any residual NaNs
X = X.fillna(X.median())

xgb = XGBRegressor(**PRUNER_XGB_PARAMS)
xgb.fit(X.values, y)

# Map booster 'f0','f1',... to feature names
booster = xgb.get_booster()
gain_raw = booster.get_score(importance_type="gain")  # keys like 'f0','f3',...

# Build mapping from f-index to feature name
feature_map = {f"f{i}": feat for i, feat in enumerate(features)}

xgb_gain = {}
for f_idx, gain_val in gain_raw.items():
    feat_name = feature_map.get(f_idx)
    if feat_name is not None:
        xgb_gain[feat_name] = gain_val

# Ensure all features have some gain value (0.0 if absent)
xgb_gain_full = {feat: xgb_gain.get(feat, 0.0) for feat in features}

# =============================================================
# STEP 4 — LINEAR EXPLAINABILITY (COEFFS + CORRS)
# =============================================================
scaler = StandardScaler()
X_std = scaler.fit_transform(X.values)

lin = LinearRegression()
lin.fit(X_std, y)
coefs = lin.coef_

corrs = []
for col in features:
    series = df[col].values
    if np.std(series) == 0:
        corrs.append(0.0)
    else:
        corrs.append(np.corrcoef(series, y)[0, 1])

corrs = np.array(corrs, dtype=float)

# =============================================================
# STEP 5 — ECONOMIC SIGN MAP
# =============================================================
# Default assumption: we don't know the sign → neutral score 0.5
economic_sign_map = {
    # Volatility memory / clustering
    "RV_14": 1,
    "RV_14_raw": 1,
    "rv_1": 1,
    "rv_5": 1,
    "rv_22": 1,
    "rv_63": 1,
    "RV_2": 1,
    "RV_3": 1,
    "RV_7": 1,
    "RV_21": 1,
    "RV_42": 1,
    "RV_126": 1,
    "RV_252": 1,
    "rv_5_22_ratio": 1,
    "rv_7_21_ratio": 1,
    "rv_21_63_ratio": 1,
    "rv_14_63_ratio": 1,
    "rv_7_126_ratio": 1,
    "RV_14_smooth_ema10": 1,
    "RV_14_deviation_from_ema10": 1,
    "ret_std_5": 1,
    "ret_std_22": 1,
    "RV_14_rollstd_21": 1,
    "vol_of_vol_14": 1,
    "vol_of_vol_30": 1,
    "vol_of_vol_ratio_14": 1,
    # Compression / range / ATR (more range → more future RV)
    "range_pct": 1,
    "gap_pct": 1,
    "bb_bandwidth": 1,      # high bandwidth = high realised vol regime
    "atr_14": 1,
    "atr_pct": 1,
    "parkinson_vol_14": 1,
    "gk_vol_14": 1,
    "rs_vol_14": 1,
    "range_pct_vol_14": 1,
    "ret_entropy_14": -1,   # higher entropy = more noise, sometimes lower predictability
    "jump_flag": 1,
    # Macro vol indices
    "vix_close": 1,
    "vxn_close": 1,
    "move_index": 1,
    "vix_vxn_spread": 1,
    "vix_over_rv14": 1,
    "vix_minus_rv14": 1,
    "move_minus_vix": 1,
    "vxn_over_vix": 1,
    # FX & credit
    "fx_vol_index_14": 1,
    "hyg_close": -1,        # higher HYG = tighter spreads / risk-on
    "dxy_index": 1,         # stronger USD often risk-off
    # Trend & momentum
    "momentum_10": 1,
    "momentum_22": 1,
    "slope_20": 1,
    "trend_strength_50": 1,
    # Oscillators (higher RSI → more stretched / often lower future RV)
    "rsi_14": -1,
    "rsi_bb_interact": -1,
    # IV / IV-RV dynamics
    "IV_14": 1,
    "IV_14_lag1": 1,
    "IV_RV_spread_lag1": 1,
    "IV_trend_5": 1,
    "IV_volatility_10": 1,
    # Microstructure / volume
    "vol_z_20": 1,
    "volume_ratio_5_20": 1,
    "turnover": 1,
    "liquidity_pressure": 1,
    "volume_volatility_20": 1,
    "price_volume_corr_20": 1,
    "signed_volume": 1,
    "ret_autocorr_1_21": 1,
}

econ_scores = []
for feat, coef_val in zip(features, coefs):
    coef_sign = np.sign(coef_val)
    expected = economic_sign_map.get(feat, None)

    if coef_sign == 0 or expected is None:
        econ_scores.append(0.5)          # neutral
    elif coef_sign == expected:
        econ_scores.append(1.0)          # aligned
    else:
        econ_scores.append(0.0)          # conflict

econ_scores = np.array(econ_scores, dtype=float)

# =============================================================
# STEP 6 — FEATURE QUALITY SCORE (0–100)
# =============================================================
coef_abs = np.abs(coefs)
max_coef = np.max(coef_abs) if np.any(np.isfinite(coef_abs)) else 0.0
if max_coef > 0:
    coef_norm = coef_abs / max_coef
else:
    coef_norm = np.zeros_like(coef_abs)

max_corr = np.nanmax(np.abs(corrs)) if np.any(np.isfinite(corrs)) else 0.0
if max_corr > 0:
    corr_norm = np.nan_to_num(np.abs(corrs) / max_corr, nan=0.0)
else:
    corr_norm = np.zeros_like(corrs)

xgb_vals = np.array([xgb_gain_full[f] for f in features], dtype=float)
max_gain = np.nanmax(xgb_vals) if np.any(np.isfinite(xgb_vals)) else 0.0
if max_gain > 0:
    xgb_norm = np.nan_to_num(xgb_vals / max_gain, nan=0.0)
else:
    xgb_norm = np.zeros_like(xgb_vals)

# Final Feature Quality Score (tweak weights here if you want)
feature_quality = 100.0 * (
    0.35 * coef_norm
    + 0.35 * corr_norm
    + 0.25 * xgb_norm
    + 0.05 * econ_scores
)

feature_scores = pd.DataFrame(
    {
        "Feature": features,
        "Coef_Standardised": coefs,
        "Corr_with_Target": corrs,
        "XGB_Gain": [xgb_gain_full[f] for f in features],
        "Economic_Score": econ_scores,
        "Feature_Quality_Score": feature_quality,
    }
).sort_values("Feature_Quality_Score", ascending=False).reset_index(drop=True)

feature_scores.to_csv(os.path.join(OUT_DIR, "feature_scores.csv"), index=False)
print("[PRUNER] Saved feature_scores.csv")

# =============================================================
# STEP 7 — FINAL SELECTION (TOP-N + THRESHOLD)
# =============================================================
topN = feature_scores.head(TOP_N_FEATURES)["Feature"].tolist()
strong = feature_scores[feature_scores["Feature_Quality_Score"] >= MIN_SCORE_THRESHOLD][
    "Feature"
].tolist()

final_features = sorted(set(topN + strong))

if len(final_features) < 3:
    print("[PRUNER] Warning: very few features selected; relaxing to Top-N only")
    final_features = sorted(set(topN))

print(f"[PRUNER] Final selected feature count : {len(final_features)}")

# Build selected_features.csv (with Included flag)
sel_df = feature_scores.copy()
sel_df["Included"] = sel_df["Feature"].isin(final_features).astype(int)
sel_df.to_csv(os.path.join(OUT_DIR, "selected_features.csv"), index=False)
print("[PRUNER] Saved selected_features.csv")

# Save plain-text list of selected features (for modelling engine)
with open(os.path.join(OUT_DIR, "selected_feature_list.txt"), "w", encoding="utf-8") as f:
    for feat in final_features:
        f.write(str(feat) + "\n")

print("[PRUNER] Saved selected_feature_list.txt")
print("[PRUNER] Pruning complete.")
