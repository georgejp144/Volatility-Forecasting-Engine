# ============================================================
# Volatility Forecasting Engine — Global Model Runner
# ============================================================
#
# PURPOSE:
#   Full global training + per-ticker forecasting pipeline.
#
#   Workflow:
#   • Load dataset and feature list
#   • Prune sparse features
#   • Train global XGB quantile models (p10, p50, p90)
#   • Train global LSTM (with time-decay weights)
#   • For each ticker:
#       – Walk-forward evaluation (XGB)
#       – Regime confidence
#       – Mahalanobis drift
#       – Ensemble forecast + spread
#       – Build dashboard .csv
#       – Return summary metrics
#
#   Output files contain:
#   • NO trading signals — forecasts + diagnostics only
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

from dotenv import load_dotenv
import os
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from sklearn.covariance import LedoitWolf
from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

# Paths
RAW_PATH = "batch_vol_runs/global_features/all_tickers_raw.parquet"
PRUNER_FEATURE_LIST = "batch_vol_runs/feature_pruner/selected_feature_list.txt"
PRUNED_FEATURE_LIST_OUT = "batch_vol_runs/feature_pruner/pruned_feature_list.txt"
OUT_ROOT = "batch_vol_runs"
SYMBOLS_CSV = "symbols.csv"
VALID_TICKERS_FILE = os.path.join(OUT_ROOT, "valid_tickers_for_training.csv")

# Core columns
TARGET_COL = "rv_14_forward"
DATE_COL = "date"
SYMBOL_COL = "symbol"

# Model timing
SEQ_WINDOW = 22
RV_WINDOW = 14
TRADING_DAYS = 252

YEARS_HISTORY = 7
LAMBDA_DECAY = 0.4

# Validation parameters
MIN_TRAIN_SIZE_WFA = 60
WFA_TEST_WINDOW = 14
RECENT_WFA_WINDOWS = 10

# Feature pruning
MAX_FEATURE_MISSINGNESS = 0.10  # per-ticker

# Determinism
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED)

load_dotenv()


# ============================================================
# DATA LOADING
# ============================================================

def load_global_dataset(raw_path=RAW_PATH):
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Could not find global feature file at: {raw_path}")

    df = pd.read_parquet(raw_path)

    if DATE_COL not in df.columns or SYMBOL_COL not in df.columns:
        raise ValueError(f"Global dataset must contain '{DATE_COL}' and '{SYMBOL_COL}'")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    if df[DATE_COL].isna().all():
        raise ValueError(f"Column '{DATE_COL}' could not be parsed as datetime.")

    if df[DATE_COL].dt.tz is not None:
        df[DATE_COL] = df[DATE_COL].dt.tz_convert(None)

    df[DATE_COL] = df[DATE_COL].dt.normalize()

    df = df.sort_values([SYMBOL_COL, DATE_COL]).reset_index(drop=True)
    return df


def load_feature_list(path=PRUNER_FEATURE_LIST):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature list not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        features = [line.strip() for line in f.readlines() if line.strip()]

    if not features:
        raise ValueError("Selected feature list is empty.")

    return features


def prune_sparse_features(df_global, features, max_missing=MAX_FEATURE_MISSINGNESS):
    valid_feats = [f for f in features if f in df_global.columns]
    if not valid_feats:
        raise RuntimeError("No requested features found in global dataset.")

    missing_frac = df_global[valid_feats].isna().mean()
    sparse_feats = missing_frac[missing_frac > max_missing].index.tolist()
    pruned_feats = [f for f in valid_feats if f not in sparse_feats]

    print(f"[FEATURES] Initial requested: {len(features)}")
    print(f"[FEATURES] Valid in dataset: {len(valid_feats)}")
    print(f"[FEATURES] Dropping {len(sparse_feats)} sparse features:")
    for sf in sparse_feats:
        print(f"   - {sf}")

    print(f"[FEATURES] Remaining: {len(pruned_feats)}")

    # Save pruned feature list
    os.makedirs(os.path.dirname(PRUNED_FEATURE_LIST_OUT), exist_ok=True)
    with open(PRUNED_FEATURE_LIST_OUT, "w", encoding="utf-8") as f:
        for feat in pruned_feats:
            f.write(feat + "\n")

    print(f"[FEATURES] Pruned list saved → {PRUNED_FEATURE_LIST_OUT}")

    return pruned_feats


# ============================================================
# GLOBAL TRAINING — XGB + LSTM
# ============================================================

def prepare_global_training(df_global, features):
    if TARGET_COL not in df_global.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'")

    df_train = df_global.dropna(subset=[TARGET_COL]).copy()

    actual_features = [c for c in features if c in df_train.columns]
    missing_feats = sorted(set(features) - set(actual_features))
    if missing_feats:
        print(f"⚠️ Missing {len(missing_feats)} features (ignored).")
        if len(actual_features) < 3:
            raise RuntimeError("Too few valid features remain.")

    features = actual_features
    feature_medians = df_train[features].median()

    X_global = df_train[features].fillna(feature_medians).values
    y_global = df_train[TARGET_COL].values

    max_date = df_train[DATE_COL].max()
    age_days = (max_date - df_train[DATE_COL]).dt.days
    age_years = age_days / 365.0
    row_weights = np.exp(-LAMBDA_DECAY * age_years).astype(float).values

    return df_train, features, feature_medians, X_global, y_global, row_weights


def train_global_xgb(X, y, weights):
    common_params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=SEED,
        tree_method="hist",
        n_jobs=1,
    )

    models = {}
    for alpha, key in [(0.1, "p10"), (0.5, "p50"), (0.9, "p90")]:
        m = XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, **common_params)
        m.fit(X, y, sample_weight=weights)
        models[key] = m

    return models


def build_global_lstm_sequences(df_train, features, feature_medians, scaler, seq_window=SEQ_WINDOW):
    X_seq, y_seq, w_seq = [], [], []

    max_date = df_train[DATE_COL].max()
    age_days = (max_date - df_train[DATE_COL]).dt.days
    age_years = age_days / 365.0
    decay_weights = np.exp(-LAMBDA_DECAY * age_years).astype(float)
    decay_weights = pd.Series(decay_weights.values, index=df_train.index)

    for symbol, df_sym in df_train.groupby(SYMBOL_COL):
        df_sym = df_sym.sort_values(DATE_COL).copy()

        X_sym_raw = df_sym[features].fillna(feature_medians)
        X_sym = scaler.transform(X_sym_raw.values)
        y_sym = df_sym[TARGET_COL].values
        indices = df_sym.index.to_list()

        if len(df_sym) <= seq_window:
            continue

        for i in range(seq_window, len(df_sym)):
            idx = indices[i]
            X_seq.append(X_sym[i-seq_window:i, :])
            y_seq.append(y_sym[i])
            w_seq.append(decay_weights.loc[idx])

    if not X_seq:
        raise RuntimeError("No LSTM sequences could be formed.")

    return np.array(X_seq), np.array(y_seq), np.array(w_seq)


def train_global_lstm(X_seq, y_seq, sample_weights, seq_window=SEQ_WINDOW):
    n = len(X_seq)
    cut = int(0.8 * n)

    X_train, y_train = X_seq[:cut], y_seq[:cut]
    X_val, y_val = X_seq[cut:], y_seq[cut:]
    w_train = sample_weights[:cut]

    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(seq_window, X_train.shape[2])),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    model.compile(optimizer="adam", loss="mae")
    es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        sample_weight=w_train,
        epochs=60,
        batch_size=32,
        callbacks=[es],
        verbose=0,
    )

    y_val_pred = model.predict(X_val, verbose=0).flatten()
    lstm_mae = mean_absolute_error(y_val, y_val_pred)

    print(f"🧠 Global LSTM trained | Validation MAE: {lstm_mae:.6f}")
    return model


def predict_lstm_distribution(model, X_seq_latest, n_samples=100):
    preds = []
    for _ in range(n_samples):
        y = model(X_seq_latest, training=True)
        preds.append(float(y.numpy().flatten()[0]))
    preds = np.array(preds)
    return np.percentile(preds, [10, 50, 90])


# ============================================================
# PER-TICKER ANALYTICS
# ============================================================

def run_wfa_xgb_for_ticker(df_sym_train, features, xgb_model, out_dir, symbol):
    n = len(df_sym_train)
    if n < (MIN_TRAIN_SIZE_WFA + WFA_TEST_WINDOW):
        print(f"⚠️ Not enough data for WFA: {symbol}")
        return None, np.nan

    os.makedirs(out_dir, exist_ok=True)

    records = []
    train_end = MIN_TRAIN_SIZE_WFA

    while train_end + WFA_TEST_WINDOW <= n:
        test_slice = slice(train_end, train_end + WFA_TEST_WINDOW)
        df_te = df_sym_train.iloc[test_slice]
        X_te = df_te[features].values
        y_te = df_te[TARGET_COL].values

        y_pred = xgb_model.predict(X_te)
        mae = mean_absolute_error(y_te, y_pred)
        relmae = (mae / max(np.mean(y_te), 1e-8)) * 100.0

        records.append({
            "Train_End_Date": df_sym_train[DATE_COL].iloc[train_end - 1],
            "Test_Start_Date": df_te[DATE_COL].iloc[0],
            "Test_End_Date": df_te[DATE_COL].iloc[-1],
            "XGB_WFA_MAE": mae,
            "XGB_WFA_RelMAE": relmae,
            "N_Test": len(y_te),
        })

        train_end += WFA_TEST_WINDOW

    if not records:
        return None, np.nan

    wfa_df = pd.DataFrame(records)
    path = os.path.join(out_dir, f"wfa_xgb_{symbol.upper()}.csv")
    wfa_df.to_csv(path, index=False)
    print(f"📈 WFA saved → {path}")

    errors = wfa_df["XGB_WFA_RelMAE"].dropna().values
    if errors.size == 0:
        return wfa_df, np.nan

    recent = errors[-RECENT_WFA_WINDOWS:] if errors.size >= RECENT_WFA_WINDOWS else errors
    recent_mean = recent.mean()

    low = np.percentile(errors, 10)
    high = np.percentile(errors, 90)
    if high <= low:
        return wfa_df, np.nan

    norm = (recent_mean - low) / (high - low)
    norm = max(0.0, min(1.0, norm))
    wfa_conf = (1.0 - norm) * 100.0

    return wfa_df, wfa_conf


def compute_regime_confidence(df_sym_train, features, scaler, wfa_df, df_sym_latest_feats):
    if wfa_df is None or wfa_df.empty:
        return np.nan

    wfa_df = wfa_df.copy()
    wfa_df["Test_Start_Date"] = pd.to_datetime(wfa_df["Test_Start_Date"])

    df_for_merge = df_sym_train[[DATE_COL] + features].copy()
    merged = pd.merge(wfa_df, df_for_merge, left_on="Test_Start_Date", right_on=DATE_COL, how="inner")

    merged = merged.dropna(subset=["XGB_WFA_RelMAE"])
    if merged.empty:
        return np.nan

    hist_feats_scaled = scaler.transform(merged[features].values)

    current_feats = df_sym_latest_feats[features].fillna(df_sym_train[features].median()).values.reshape(1, -1)
    current_feats_scaled = scaler.transform(current_feats)[0]

    diffs = hist_feats_scaled - current_feats_scaled
    dists = np.linalg.norm(diffs, axis=1)

    K = min(10, len(merged))
    idx = np.argsort(dists)[:K]
    similar_errors = merged.iloc[idx]["XGB_WFA_RelMAE"].values

    if similar_errors.size == 0:
        return np.nan

    similar_mean = similar_errors.mean()
    all_errors = merged["XGB_WFA_RelMAE"].values
    low = np.percentile(all_errors, 10)
    high = np.percentile(all_errors, 90)
    if high <= low:
        return np.nan

    norm = (similar_mean - low) / (high - low)
    norm = max(0.0, min(1.0, norm))
    return (1.0 - norm) * 100.0


def compute_mahalanobis_drift(df_sym_train, df_sym_full, features):
    try:
        feat_train = df_sym_train[features].copy().fillna(df_sym_train[features].median())

        if len(feat_train) <= len(features):
            return np.nan, None

        lw = LedoitWolf().fit(feat_train.values)
        cov_reg = lw.covariance_
        inv_cov = np.linalg.inv(cov_reg)
        mu = feat_train.mean(axis=0)

        feat_all = df_sym_full[features].copy().fillna(mu)
        diffs = feat_all.values - mu.values
        d_mahal = np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs))

        med = np.nanmedian(d_mahal)
        if med == 0 or np.isnan(med):
            return np.nan, None

        drift_scores = d_mahal / med
        return float(drift_scores[-1]), drift_scores

    except Exception as e:
        print(f"⚠️ Drift error: {e}")
        return np.nan, None


def drift_reliability_from_score(drift_score):
    if drift_score is None or np.isnan(drift_score):
        return np.nan
    if drift_score <= 1.0:
        return 100.0
    if drift_score >= 4.0:
        return 0.0
    return 100.0 * (4.0 - drift_score) / 3.0


def compute_model_reliability_score(wfa_conf, regime_conf, drift_score):
    if any([
        wfa_conf is None, np.isnan(wfa_conf),
        regime_conf is None, np.isnan(regime_conf),
        drift_score is None, np.isnan(drift_score)
    ]):
        return np.nan

    drift_rel = drift_reliability_from_score(drift_score)
    if np.isnan(drift_rel):
        return np.nan

    mrs = (
        0.4 * float(wfa_conf)
        + 0.4 * float(regime_conf)
        + 0.2 * float(drift_rel)
    )
    return max(0.0, min(100.0, mrs))


# ============================================================
# DASHBOARD + METRICS FOR A SINGLE TICKER
# ============================================================

def build_dashboard_for_ticker(
    df_sym_full,
    features,
    xgb_models,
    lstm_model,
    scaler,
    feature_medians,
    wfa_df,
    wfa_confidence,
    regime_confidence,
    drift_score,
    symbol,
    out_root=OUT_ROOT,
):
    # Guarantee missing base columns exist
    for col in ["RV_14", "IV_14", "close", "bb_bandwidth", "atr_pct", "atr_14"]:
        if col not in df_sym_full.columns:
            df_sym_full[col] = np.nan

    # Training slice
    df_sym_train = df_sym_full.dropna(subset=[TARGET_COL]).copy()

    if df_sym_train.empty:
        raise RuntimeError(f"No training data for {symbol}")

    df_sym_train[features] = df_sym_train[features].fillna(feature_medians)

    # Evaluate XGB MAE (simple 80/20)
    xgb_median = xgb_models["p50"]
    cut = int(len(df_sym_train) * 0.8)
    df_train, df_test = df_sym_train.iloc[:cut], df_sym_train.iloc[cut:]

    if len(df_test) > 0:
        y_pred = xgb_median.predict(df_test[features].values)
        mae = mean_absolute_error(df_test[TARGET_COL].values, y_pred)
        xgb_relmae = (mae / max(np.mean(df_test[TARGET_COL]), 1e-8)) * 100.0
    else:
        xgb_relmae = 100.0

    # LSTM per-ticker validation
    X_sym = df_sym_train[features].values
    X_sym_scaled = scaler.transform(X_sym)
    y_sym = df_sym_train[TARGET_COL].values

    X_seq_list, y_seq_list = [], []
    for i in range(SEQ_WINDOW, len(df_sym_train)):
        X_seq_list.append(X_sym_scaled[i - SEQ_WINDOW:i, :])
        y_seq_list.append(y_sym[i])

    if len(X_seq_list) > 1:
        cutoff = int(len(X_seq_list) * 0.8)
        X_train_seq = np.array(X_seq_list[:cutoff])
        X_test_seq = np.array(X_seq_list[cutoff:])
        y_test_seq = np.array(y_seq_list[cutoff:])
        if len(X_test_seq) == 0:
            X_test_seq = np.array(X_seq_list)
            y_test_seq = np.array(y_seq_list)
        lstm_pred = lstm_model.predict(X_test_seq, verbose=0).flatten()
        lstm_mae = mean_absolute_error(y_test_seq, lstm_pred)
        lstm_relmae = (lstm_mae / max(np.mean(y_test_seq), 1e-8)) * 100.0
    else:
        lstm_relmae = xgb_relmae

    # Latest valid row for forecasting
    df_latest_feats = df_sym_full.dropna(subset=features).copy()
    if df_latest_feats.empty:
        raise RuntimeError(f"No valid feature row for {symbol}")

    latest_row = df_latest_feats.iloc[-1]
    latest_date = latest_row[DATE_COL]
    candidates = df_sym_full.index[df_sym_full[DATE_COL] == latest_date]
    latest_index = int(candidates.max()) if len(candidates) else int(df_sym_full.index[-1])

    # XGB quantile forecasts
    X_latest = latest_row[features].fillna(feature_medians).values.reshape(1, -1)

    p10, p50, p90 = [float(x.predict(X_latest)[0]) * 100.0 for x in [
        xgb_models["p10"], xgb_models["p50"], xgb_models["p90"]
    ]]

    # LSTM quantiles via MC-dropout
    seq_candidates = df_sym_full.copy()
    seq_candidates[features] = seq_candidates[features].fillna(feature_medians)
    X_seq_latest = scaler.transform(seq_candidates.iloc[-SEQ_WINDOW:][features].values).reshape(1, SEQ_WINDOW, -1)

    lp10, lp50, lp90 = [v * 100.0 for v in predict_lstm_distribution(lstm_model, X_seq_latest, 100)]

    # Use p50 as main forecasts
    xgb_forecast_pct = p50
    lstm_forecast_pct = lp50

    # Current RV
    current_rv_pct = float(df_sym_full["RV_14"].iloc[latest_index]) if "RV_14" in df_sym_full.columns else np.nan

    # Ensemble weighting
    lstm_weight = 1.0 / max(lstm_relmae, 1e-8)
    xgb_weight = 1.0 / max(xgb_relmae, 1e-8)

    # WFA scaling
    if np.isfinite(wfa_confidence):
        wfa_norm = np.clip(wfa_confidence / 100.0, 0.0, 1.0)
    else:
        wfa_norm = 0.5

    lstm_wfa_score = max(1.0 - wfa_norm, 0.1)
    xgb_wfa_score = max(wfa_norm, 0.1)

    # Regime scaling
    if np.isfinite(regime_confidence):
        reg_norm = np.clip(regime_confidence / 100.0, 0.0, 1.0)
    else:
        reg_norm = 0.5

    lstm_reg_score = max(1.0 - reg_norm, 0.1)
    xgb_reg_score = max(reg_norm, 0.1)

    # Drift factor
    if np.isfinite(drift_score):
        drift_rel_100 = drift_reliability_from_score(drift_score)
        drift_factor = max(drift_rel_100 / 100.0, 0.1)
    else:
        drift_factor = 1.0

    w_lstm = lstm_weight * lstm_wfa_score * lstm_reg_score * drift_factor
    w_xgb = xgb_weight * xgb_wfa_score * xgb_reg_score * drift_factor

    denom = w_lstm + w_xgb
    if denom > 0:
        lstm_weight = w_lstm / denom
        xgb_weight = w_xgb / denom
    else:
        lstm_weight = lstm_weight / (lstm_weight + xgb_weight)
        xgb_weight = xgb_weight / (lstm_weight + xgb_weight)

    # Ensemble quantiles
    ensemble_p10 = lstm_weight * lp10 + xgb_weight * p10
    ensemble_p50 = lstm_weight * lstm_forecast_pct + xgb_weight * xgb_forecast_pct
    ensemble_p90 = lstm_weight * lp90 + xgb_weight * p90
    ensemble_spread = ensemble_p90 - ensemble_p10

    # IV/RV
    iv14 = float(df_sym_full["IV_14"].iloc[latest_index]) if "IV_14" in df_sym_full.columns else np.nan
    rv14 = float(df_sym_full["RV_14"].iloc[latest_index]) if "RV_14" in df_sym_full.columns else np.nan

    iv_minus_rv = iv14 - rv14 if np.isfinite(iv14) and np.isfinite(rv14) else np.nan
    forecast_vs_iv = iv14 - lstm_forecast_pct if np.isfinite(iv14) else np.nan
    ensemble_vs_iv = iv14 - ensemble_p50 if np.isfinite(iv14) else np.nan
    xgb_vs_iv = iv14 - xgb_forecast_pct if np.isfinite(iv14) else np.nan

    # Del
    delta_lstm = lstm_forecast_pct - current_rv_pct if np.isfinite(current_rv_pct) else np.nan
    delta_xgb = xgb_forecast_pct - current_rv_pct if np.isfinite(current_rv_pct) else np.nan

    # Compression diagnostics
    df_sym_full["bb_bandwidth_pctile"] = df_sym_full["bb_bandwidth"].rank(pct=True) * 100
    df_sym_full["atr_pct_pctile"] = df_sym_full["atr_pct"].rank(pct=True) * 100

    bb_band_pct = float(df_sym_full["bb_bandwidth_pctile"].iloc[latest_index])
    atr_pct_pct = float(df_sym_full["atr_pct_pctile"].iloc[latest_index])

    # MRS
    mrs_score = compute_model_reliability_score(wfa_confidence, regime_confidence, drift_score)

    # Error diagnostics
    ensemble_scaled_mae = ensemble_normalized_rmse = ensemble_iv_rel_error = np.nan

    if np.isfinite(current_rv_pct) and np.isfinite(ensemble_p50):
        err = abs(ensemble_p50 - current_rv_pct)

        if current_rv_pct > 0:
            ensemble_scaled_mae = err / current_rv_pct

        regime_rv = df_sym_full["RV_14"].tail(63).mean()
        if np.isfinite(regime_rv) and regime_rv > 0:
            ensemble_normalized_rmse = err / regime_rv

        if np.isfinite(iv14) and iv14 > 0:
            ensemble_iv_rel_error = err / iv14

    # Attach outputs
    for c in [
        "Forecast_RV_14", "Forecast_RV_14_p10", "Forecast_RV_14_p90",
        "XGB_Forecast_RV_14", "XGB_Forecast_RV_14_p10", "XGB_Forecast_RV_14_p90",
        "Ensemble_Forecast_RV_14", "Ensemble_Forecast_RV_14_p10", "Ensemble_Forecast_RV_14_p90",
        "Ensemble_Forecast_RV_14_Spread", "LSTM_Weight", "XGB_Weight",
        "Relative_MAE_LSTM", "Relative_MAE_XGB",
        "IV_minus_RV", "Forecast_vs_IV", "XGB_vs_IV", "Ensemble_vs_IV",
        "Model_Disagreement", "LSTM_Delta_RV", "XGB_Delta_RV",
        "XGB_WFA_Confidence", "XGB_Regime_Confidence",
        "Mahalanobis_Drift_Score", "Model_Reliability_Score",
        "Ensemble_Scaled_MAE", "Ensemble_Normalized_RMSE", "Ensemble_IV_Relative_Error",
    ]:
        if c not in df_sym_full.columns:
            df_sym_full[c] = np.nan

    df_sym_full.loc[latest_index, [
        "Forecast_RV_14", "Forecast_RV_14_p10", "Forecast_RV_14_p90",
        "XGB_Forecast_RV_14", "XGB_Forecast_RV_14_p10", "XGB_Forecast_RV_14_p90",
        "Ensemble_Forecast_RV_14", "Ensemble_Forecast_RV_14_p10", "Ensemble_Forecast_RV_14_p90",
        "Ensemble_Forecast_RV_14_Spread",
        "LSTM_Weight", "XGB_Weight",
        "Relative_MAE_LSTM", "Relative_MAE_XGB",
        "IV_minus_RV", "Forecast_vs_IV", "XGB_vs_IV", "Ensemble_vs_IV",
        "Model_Disagreement", "LSTM_Delta_RV", "XGB_Delta_RV",
        "XGB_WFA_Confidence", "XGB_Regime_Confidence",
        "Mahalanobis_Drift_Score", "Model_Reliability_Score",
        "Ensemble_Scaled_MAE", "Ensemble_Normalized_RMSE", "Ensemble_IV_Relative_Error",
    ]] = [
        lp50, lp10, lp90,
        p50, p10, p90,
        ensemble_p50, ensemble_p10, ensemble_p90,
        ensemble_spread,
        lstm_weight, xgb_weight,
        lstm_relmae, xgb_relmae,
        iv_minus_rv, forecast_vs_iv, xgb_vs_iv, ensemble_vs_iv,
        abs(lstm_forecast_pct - xgb_forecast_pct),
        delta_lstm, delta_xgb,
        wfa_confidence, regime_confidence,
        drift_score, mrs_score,
        ensemble_scaled_mae, ensemble_normalized_rmse, ensemble_iv_rel_error,
    ]

    # Dashboard output
    dashboard_cols = [
        DATE_COL, "close", "RV_14", "IV_14", "IV_minus_RV",
        "Forecast_RV_14", "Forecast_RV_14_p10", "Forecast_RV_14_p90",
        "Forecast_vs_IV",
        "XGB_Forecast_RV_14", "XGB_Forecast_RV_14_p10", "XGB_Forecast_RV_14_p90",
        "XGB_vs_IV",
        "Ensemble_Forecast_RV_14", "Ensemble_Forecast_RV_14_p10",
        "Ensemble_Forecast_RV_14_p90", "Ensemble_Forecast_RV_14_Spread",
        "Ensemble_vs_IV",
        "LSTM_Weight", "XGB_Weight",
        "Relative_MAE_LSTM", "Relative_MAE_XGB",
        "Model_Disagreement", "LSTM_Delta_RV", "XGB_Delta_RV",
        "bb_bandwidth", "bb_bandwidth_pctile",
        "atr_14", "atr_pct", "atr_pct_pctile",
        "XGB_WFA_Confidence", "XGB_Regime_Confidence",
        "Mahalanobis_Drift_Score", "Model_Reliability_Score",
        "Ensemble_Scaled_MAE", "Ensemble_Normalized_RMSE",
        "Ensemble_IV_Relative_Error",
    ]

    for c in dashboard_cols:
        if c not in df_sym_full.columns:
            df_sym_full[c] = np.nan

    out_df = df_sym_full[dashboard_cols].dropna(subset=["RV_14"])
    os.makedirs(out_root, exist_ok=True)

    out_path = os.path.join(out_root, f"vol_dashboard_data_{symbol.upper()}.csv")
    out_df.to_csv(out_path, index=False)

    print(f"✅ Dashboard saved → {out_path}")

    latest_row_dash = out_df.iloc[-1]

    summary_rec = {
        "symbol": symbol.upper(),
        "date": latest_row_dash[DATE_COL],
        "IV_14": latest_row_dash["IV_14"],
        "RV_14": latest_row_dash["RV_14"],
        "Ensemble_Forecast_RV_14": latest_row_dash["Ensemble_Forecast_RV_14"],
        "Ensemble_Forecast_RV_14_p10": latest_row_dash["Ensemble_Forecast_RV_14_p10"],
        "Ensemble_Forecast_RV_14_p90": latest_row_dash["Ensemble_Forecast_RV_14_p90"],
        "Ensemble_Forecast_RV_14_Spread": latest_row_dash["Ensemble_Forecast_RV_14_Spread"],
        "Ensemble_vs_IV": latest_row_dash["Ensemble_vs_IV"],
        "LSTM_Weight": latest_row_dash["LSTM_Weight"],
        "XGB_Weight": latest_row_dash["XGB_Weight"],
        "Relative_MAE_LSTM": latest_row_dash["Relative_MAE_LSTM"],
        "Relative_MAE_XGB": latest_row_dash["Relative_MAE_XGB"],
        "Ensemble_Scaled_MAE": latest_row_dash["Ensemble_Scaled_MAE"],
        "Ensemble_Normalized_RMSE": latest_row_dash["Ensemble_Normalized_RMSE"],
        "Ensemble_IV_Relative_Error": latest_row_dash["Ensemble_IV_Relative_Error"],
        "XGB_WFA_Confidence": latest_row_dash["XGB_WFA_Confidence"],
        "XGB_Regime_Confidence": latest_row_dash["XGB_Regime_Confidence"],
        "Mahalanobis_Drift_Score": latest_row_dash["Mahalanobis_Drift_Score"],
        "Model_Reliability_Score": latest_row_dash["Model_Reliability_Score"],
    }

    return summary_rec


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    os.makedirs(OUT_ROOT, exist_ok=True)

    # Load dataset
    df_global = load_global_dataset(RAW_PATH)
    FEATURES_RAW = load_feature_list(PRUNER_FEATURE_LIST)

    print(f"[GLOBAL] Loaded dataset: {df_global.shape[0]} rows, {df_global.shape[1]} columns")

    # Prune sparse features
    FEATURES = prune_sparse_features(df_global, FEATURES_RAW)
    print(f"[GLOBAL] Using {len(FEATURES)} features")

    # Prepare for training
    df_train_global, FEATURES, feature_medians, X_global, y_global, row_weights_global = \
        prepare_global_training(df_global, FEATURES)

    print(f"[GLOBAL] Training rows: {len(df_train_global)}")

    # Train XGB models
    xgb_models = train_global_xgb(X_global, y_global, row_weights_global)
    print("🌳 Trained XGB quantile models")

    # Scaler
    df_train_filled = df_train_global[FEATURES].fillna(feature_medians)
    scaler = RobustScaler()
    scaler.fit(df_train_filled.values)

    # LSTM sequences + training
    X_seq_global, y_seq_global, w_seq_global = build_global_lstm_sequences(
        df_train_global, FEATURES, feature_medians, scaler, SEQ_WINDOW
    )
    lstm_model = train_global_lstm(X_seq_global, y_seq_global, w_seq_global, SEQ_WINDOW)

    # Determine ticker universe
    if os.path.exists(VALID_TICKERS_FILE):
        tickers = (
            pd.read_csv(VALID_TICKERS_FILE, header=None)
            .iloc[:, 0]
            .astype(str)
            .str.strip()
            .dropna()
            .tolist()
        )
        print(f"[GLOBAL] Using validated tickers → {VALID_TICKERS_FILE}")
    elif os.path.exists(SYMBOLS_CSV):
        tickers = (
            pd.read_csv(SYMBOLS_CSV)
            .iloc[:, 0]
            .astype(str)
            .str.strip()
            .dropna()
            .tolist()
        )
        print(f"[GLOBAL] Loaded tickers from {SYMBOLS_CSV}")
    else:
        tickers = sorted(df_global[SYMBOL_COL].dropna().unique().tolist())
        print("[GLOBAL] Inferred ticker universe")

    print(f"[GLOBAL] Ticker universe: {tickers}")

    summary_records = []

    for sym in tickers:
        df_sym = df_global[df_global[SYMBOL_COL] == sym].copy()

        if df_sym.empty:
            print(f"❌ {sym}: no rows")
            continue

        print(f"\n=== {sym} ===")

        df_sym = df_sym.sort_values(DATE_COL).reset_index(drop=True)

        df_sym_train = df_sym.dropna(subset=[TARGET_COL]).copy()
        if len(df_sym_train) < (MIN_TRAIN_SIZE_WFA + WFA_TEST_WINDOW):
            print(f"⚠️ {sym}: insufficient data for full WFA")

        df_sym_train[FEATURES] = df_sym_train[FEATURES].fillna(feature_medians)

        wfa_df, wfa_conf = run_wfa_xgb_for_ticker(
            df_sym_train, FEATURES, xgb_models["p50"], OUT_ROOT, sym
        )

        latest_feats = df_sym_train.iloc[[-1]] if not df_sym_train.empty else df_sym.iloc[[-1]]
        regime_conf = compute_regime_confidence(
            df_sym_train, FEATURES, scaler, wfa_df, latest_feats
        )

        drift_score, _ = compute_mahalanobis_drift(df_sym_train, df_sym, FEATURES)

        try:
            rec = build_dashboard_for_ticker(
                df_sym_full=df_sym,
                features=FEATURES,
                xgb_models=xgb_models,
                lstm_model=lstm_model,
                scaler=scaler,
                feature_medians=feature_medians,
                wfa_df=wfa_df,
                wfa_confidence=wfa_conf,
                regime_confidence=regime_conf,
                drift_score=drift_score,
                symbol=sym,
                out_root=OUT_ROOT,
            )
            summary_records.append(rec)

        except Exception as e:
            print(f"❌ {sym}: {e}")

    if summary_records:
        summary_df = pd.DataFrame(summary_records)
        path = os.path.join(OUT_ROOT, "summary_metrics.csv")
        summary_df.to_csv(path, index=False)
        print(f"\n📊 Summary saved → {path}\n")
    else:
        print("\n⚠️ No summary records produced.\n")
