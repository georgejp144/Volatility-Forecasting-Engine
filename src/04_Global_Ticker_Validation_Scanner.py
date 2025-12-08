# ============================================================
# Volatility Forecasting Engine — Ticker Validation Scanner
# ============================================================
#
# PURPOSE:
#   Validate each ticker for model readiness:
#     • Sufficient target rows for WFA
#     • Feature missingness per-ticker
#     • LSTM sequence viability
#     • Latest-row completeness
#     • Stable covariance (Ledoit–Wolf)
#     • IV_14 and RV_14 availability
#
# INPUTS:
#   batch_vol_runs/global_features/all_tickers_raw.parquet
#   pruned_feature_list.txt (preferred)
#   selected_feature_list.txt (fallback)
#   symbols.csv (optional universe)
#
# OUTPUTS:
#   batch_vol_runs/ticker_validation_report.csv
#   batch_vol_runs/valid_tickers_for_training.csv
#
# NOTES:
#   • This does not train models — only validates data
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd

from sklearn.covariance import LedoitWolf
from numpy.linalg import LinAlgError


# ============================================================
# CONFIGURATION
# ============================================================

RAW_PATH = "batch_vol_runs/global_features/all_tickers_raw.parquet"

PRUNED_FEATURE_LIST = "batch_vol_runs/feature_pruner/pruned_feature_list.txt"
SELECTED_FEATURE_LIST = "batch_vol_runs/feature_pruner/selected_feature_list.txt"

SYMBOLS_CSV = "symbols.csv"

OUT_VALID_TICKERS = "batch_vol_runs/valid_tickers_for_training.csv"
OUT_VALIDATION_REPORT = "batch_vol_runs/ticker_validation_report.csv"

TARGET_COL = "rv_14_forward"
DATE_COL = "date"
SYMBOL_COL = "symbol"

SEQ_WINDOW = 22
MIN_TRAIN_SIZE_WFA = 60
WFA_TEST_WINDOW = 14

MAX_FEATURE_MISSINGNESS = 0.10  # Per-ticker missingness threshold


# ============================================================
# GLOBAL DATASET LOADER
# ============================================================

def load_global_dataset():
    """
    Loads the entire global feature dataset and sorts by symbol/date.
    """
    df = pd.read_parquet(RAW_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.normalize()
    df = df.sort_values([SYMBOL_COL, DATE_COL]).reset_index(drop=True)
    return df


# ============================================================
# FEATURE LIST LOADER
# ============================================================

def load_feature_list():
    """
    Use pruned_feature_list.txt if available, else fallback to selected_feature_list.txt.
    """
    if os.path.exists(PRUNED_FEATURE_LIST):
        print(f"[SCAN] Using pruned feature list: {PRUNED_FEATURE_LIST}")
        path = PRUNED_FEATURE_LIST
    else:
        print(f"[SCAN] Using selected feature list: {SELECTED_FEATURE_LIST}")
        path = SELECTED_FEATURE_LIST

    with open(path, "r") as f:
        feats = [x.strip() for x in f.readlines() if x.strip()]
    return feats


# ============================================================
# LEDOIT–WOLF COVARIANCE CHECK
# ============================================================

def covariance_is_stable(df_feat):
    """
    Returns: (is_stable, condition_number)
    """
    try:
        lw = LedoitWolf().fit(df_feat.values)
        cov = lw.covariance_

        cond_number = np.linalg.cond(cov)
        return cond_number < 1e10, cond_number

    except LinAlgError:
        return False, np.inf
    except Exception:
        return False, np.inf


# ============================================================
# PER-TICKER VALIDATION LOGIC
# ============================================================

def validate_ticker(df_sym, features):
    diagnostics = {}

    # --------------------------------------------------------
    # A: TARGET AVAILABILITY (required for WFA)
    # --------------------------------------------------------
    df_train = df_sym.dropna(subset=[TARGET_COL])
    target_rows = len(df_train)
    diagnostics["target_rows"] = target_rows

    if target_rows < (MIN_TRAIN_SIZE_WFA + WFA_TEST_WINDOW):
        return False, "Insufficient rows for WFA", diagnostics

    # --------------------------------------------------------
    # B: FEATURE MISSINGNESS PER TICKER
    # --------------------------------------------------------
    missingness = df_sym[features].isna().mean()
    max_missing = float(missingness.max())
    diagnostics["feature_missingness_max"] = max_missing

    if max_missing > MAX_FEATURE_MISSINGNESS:
        return False, "Per-ticker feature missingness too high", diagnostics

    # --------------------------------------------------------
    # C: LSTM SEQUENCE VIABILITY
    # --------------------------------------------------------
    df_complete = df_sym.dropna(subset=features)
    complete_rows = len(df_complete)
    diagnostics["complete_feature_rows"] = complete_rows

    if complete_rows < SEQ_WINDOW:
        return False, "Insufficient complete rows for LSTM", diagnostics

    # --------------------------------------------------------
    # D: LATEST ROW COMPLETENESS
    # --------------------------------------------------------
    latest_row = df_sym.iloc[-1]
    missing_latest = int(latest_row[features].isna().sum())
    diagnostics["missing_features_on_latest_row"] = missing_latest

    if missing_latest > 0:
        return False, "Missing features on latest row", diagnostics

    # --------------------------------------------------------
    # E: STABLE COVARIANCE (Ledoit–Wolf)
    # --------------------------------------------------------
    df_train_feats = df_train[features].dropna()
    cov_ok, cond_number = covariance_is_stable(df_train_feats)

    diagnostics["covariance_stable"] = bool(cov_ok)
    diagnostics["covariance_condition_number"] = float(cond_number)

    if not cov_ok:
        return False, "Covariance matrix unstable (LW)", diagnostics

    # --------------------------------------------------------
    # F: MUST HAVE IV_14 AND RV_14 ON LATEST ROW
    # --------------------------------------------------------
    latest_iv_missing = pd.isna(latest_row["IV_14"]) if "IV_14" in df_sym.columns else True
    latest_rv_missing = pd.isna(latest_row["RV_14"]) if "RV_14" in df_sym.columns else True

    diagnostics["missing_IV_RV_latest"] = int(latest_iv_missing or latest_rv_missing)

    if latest_iv_missing or latest_rv_missing:
        return False, "Missing IV_14 or RV_14 on latest row", diagnostics

    # --------------------------------------------------------
    # PASSED EVERYTHING
    # --------------------------------------------------------
    return True, "Valid", diagnostics


# ============================================================
# MAIN SCAN
# ============================================================

if __name__ == "__main__":
    print("🔍 Running Ticker Validation Scanner V2.1...")

    df_global = load_global_dataset()
    FEATURES = load_feature_list()

    # Determine universe
    if os.path.exists(SYMBOLS_CSV):
        symbols = (
            pd.read_csv(SYMBOLS_CSV)
            .iloc[:, 0]
            .astype(str)
            .str.strip()
            .tolist()
        )
    else:
        symbols = sorted(df_global[SYMBOL_COL].unique().tolist())

    results = []

    # --------------------------------------------------------
    # PER-SYMBOL VALIDATION LOOP
    # --------------------------------------------------------
    for sym in symbols:
        df_sym = df_global[df_global[SYMBOL_COL] == sym].copy()

        if df_sym.empty:
            results.append({
                "symbol": sym,
                "model_ready": False,
                "reason": "No rows",
            })
            print(f"{sym}: FAIL — No rows")
            continue

        valid, reason, diag = validate_ticker(df_sym, FEATURES)
        print(f"{sym}: {'OK' if valid else 'FAIL'} — {reason}")

        results.append({
            "symbol": sym,
            "model_ready": valid,
            "reason": reason,
            **diag
        })

    # --------------------------------------------------------
    # SAVE OUTPUTS
    # --------------------------------------------------------
    report_df = pd.DataFrame(results)
    report_df.to_csv(OUT_VALIDATION_REPORT, index=False)

    valid_list = report_df[report_df["model_ready"] == True]["symbol"]
    valid_list.to_csv(OUT_VALID_TICKERS, index=False, header=False)

    print("\n📄 Saved:", OUT_VALIDATION_REPORT)
    print("📄 Saved:", OUT_VALID_TICKERS)
    print("\n✅ Validation scan completed.\n")
