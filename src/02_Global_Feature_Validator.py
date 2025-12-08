# ============================================================
# Volatility Forecasting Engine — Data Validation Module
#
# PURPOSE:
#   Institutional-grade validation for:
#     1) Per-ticker raw feature files
#     2) Unified global raw feature file
#
# VALIDATION LAYERS:
#   (1) Per-Ticker Validation
#       - Checks:
#           • Date integrity
#           • OHLCV geometry
#           • Macro ranges
#           • RV/IV sanity
#           • Tail risk / jumps
#           • NaN floods
#
#   (2) Global Validation
#       - Checks:
#           • Duplicate rows
#           • Short histories
#           • Date alignment
#           • Cross-ticker divergences
#           • Target leakage
#
# OUTPUTS:
#   batch_vol_runs/validation/
#       validation_report_<SYMBOL>.txt
#       validation_report_<SYMBOL>.json
#       global_validation.txt
#       global_validation.json
#       consolidated_summary.txt
#       consolidated_summary.json
#
# RUN:
#   02_Global_Feature_Validator.py
#
# NOTES:
#   - Safe for batch runs, missing data detection baked-in
#   - Health scores summarise warnings/critical failures
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import json
import numpy as np
import pandas as pd


# ============================================================
# PATH CONFIG
# ============================================================

RAW_DIR = "batch_vol_runs/global_features/raw_per_ticker"
GLOBAL_PATH = "batch_vol_runs/global_features/all_tickers_raw.parquet"

OUT_DIR = "batch_vol_runs/validation"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def rolling_z(df: pd.DataFrame, col: str, window: int = 90) -> pd.Series:
    """Rolling z-score for anomaly detection."""
    s = df[col].astype(float)
    mu = s.rolling(window).mean()
    sd = s.rolling(window).std()
    return (s - mu) / sd


def add_issue(issues: list, level: str, msg: str):
    """Append structured issue dict to issues list."""
    issues.append({"level": level, "message": msg})


# ============================================================
# PER-TICKER VALIDATION
# ============================================================

def validate_ticker(df: pd.DataFrame, symbol: str):
    issues = []

    # ---------------------------------------------------------
    # 1. Date integrity
    # ---------------------------------------------------------
    if "date" not in df.columns:
        add_issue(issues, "CRITICAL", "Missing 'date' column.")
        return issues

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        add_issue(issues, "CRITICAL", "Duplicated dates detected.")

    expected_days = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    if len(expected_days) != len(df):
        add_issue(
            issues, "WARNING",
            f"Missing calendar days (expected {len(expected_days)} got {len(df)})."
        )

    # ---------------------------------------------------------
    # 2. OHLCV geometry & integrity
    # ---------------------------------------------------------
    req = ["open", "high", "low", "close", "volume"]
    for c in req:
        if c not in df.columns:
            add_issue(issues, "CRITICAL", f"Missing required column {c}")
            return issues

    if (df["close"] <= 0).any():
        add_issue(issues, "CRITICAL", "Close price <= 0")

    if (df["high"] < df["low"]).any():
        add_issue(issues, "CRITICAL", "High < Low")

    if (df["volume"] < 0).any():
        add_issue(issues, "CRITICAL", "Negative volume")

    bad_candle = (
        (df["open"] < df["low"]) | (df["open"] > df["high"]) |
        (df["close"] < df["low"]) | (df["close"] > df["high"])
    )
    if bad_candle.any():
        add_issue(
            issues, "CRITICAL",
            f"Candle geometry violations ({bad_candle.sum()} rows)."
        )

    # ---------------------------------------------------------
    # 3. Return sanity
    # ---------------------------------------------------------
    if "ret" in df.columns:
        big = df["ret"].abs() > 0.20
        if big.any():
            add_issue(issues, "WARNING", f"Returns >20% ({big.sum()} rows).")

    # ---------------------------------------------------------
    # 4. Gap sanity
    # ---------------------------------------------------------
    if "prev_close" in df.columns:
        gap = (df["open"] - df["prev_close"]).abs() / df["prev_close"].replace(0, np.nan)
        g = gap > 0.15
        if g.any():
            add_issue(issues, "WARNING", f"Gaps >15% ({g.sum()} rows).")

    # ---------------------------------------------------------
    # 5. Macro ranges
    # ---------------------------------------------------------
    if "vix_close" in df.columns:
        if ((df["vix_close"] < 1) | (df["vix_close"] > 200)).any():
            add_issue(issues, "WARNING", "VIX outside 1–200.")

    if "move_index" in df.columns:
        if ((df["move_index"] < 10) | (df["move_index"] > 400)).any():
            add_issue(issues, "WARNING", "MOVE outside 10–400.")

    if "fx_gbpusd" in df.columns:
        if ((df["fx_gbpusd"] < 0.5) | (df["fx_gbpusd"] > 2.5)).any():
            add_issue(issues, "WARNING", "GBPUSD abnormal.")

    if "fx_eurusd" in df.columns:
        if ((df["fx_eurusd"] < 0.5) | (df["fx_eurusd"] > 2.0)).any():
            add_issue(issues, "WARNING", "EURUSD abnormal.")

    if "fx_jpyusd" in df.columns:
        if ((df["fx_jpyusd"] < 50) | (df["fx_jpyusd"] > 200)).any():
            add_issue(issues, "WARNING", "USD/JPY abnormal.")

    if "oil" in df.columns:
        if ((df["oil"] < 0) | (df["oil"] > 500)).any():
            add_issue(issues, "WARNING", "Oil abnormal.")

    if "gold" in df.columns:
        if ((df["gold"] < 200) | (df["gold"] > 5000)).any():
            add_issue(issues, "WARNING", "Gold abnormal.")

    # ---------------------------------------------------------
    # 6. RV & IV sanity
    # ---------------------------------------------------------
    if "RV_14" in df.columns:
        if (df["RV_14"] < 0).any():
            add_issue(issues, "CRITICAL", "Negative RV_14.")

        z_rv = rolling_z(df, "RV_14")
        if (z_rv.abs() > 5).any():
            add_issue(issues, "WARNING", "RV_14 z-scores >5.")

    if "IV_14" in df.columns:
        if ((df["IV_14"] < 1) | (df["IV_14"] > 300)).any():
            add_issue(issues, "WARNING", "IV_14 outside 1–300.")

        z_iv = rolling_z(df, "IV_14")
        if (z_iv.abs() > 5).any():
            add_issue(issues, "WARNING", "IV_14 z-scores >5.")

    # ---------------------------------------------------------
    # 7. Vol-of-vol
    # ---------------------------------------------------------
    if "vol_of_vol_14" in df.columns:
        if (df["vol_of_vol_14"] < 0).any():
            add_issue(issues, "CRITICAL", "Negative vol_of_vol_14.")

    # ---------------------------------------------------------
    # 8. Tail risk
    # ---------------------------------------------------------
    if "realised_kurtosis_14" in df.columns:
        if (df["realised_kurtosis_14"] < 1).any():
            add_issue(issues, "WARNING", "Kurtosis <1.")

    if "realised_skew_14" in df.columns:
        if (df["realised_skew_14"].abs() > 5).any():
            add_issue(issues, "WARNING", "Skew >5.")

    # ---------------------------------------------------------
    # 9. Trend & volume
    # ---------------------------------------------------------
    if "adx_14" in df.columns:
        if ((df["adx_14"] < 0) | (df["adx_14"] > 100)).any():
            add_issue(issues, "WARNING", "ADX_14 abnormal.")

    if "volume" in df.columns:
        z_vol = rolling_z(df, "volume")
        if (z_vol.abs() > 6).any():
            add_issue(issues, "WARNING", "Volume z-scores >6.")

    # ---------------------------------------------------------
    # 10. NaN floods
    # ---------------------------------------------------------
    nan_frac = df.isna().mean()
    high_nan = nan_frac[nan_frac > 0.2]
    for col, frac in high_nan.items():
        add_issue(issues, "WARNING", f"{col} is {frac*100:.1f}% NaN.")

    return issues


# ============================================================
# GLOBAL VALIDATION
# ============================================================

def validate_global(df: pd.DataFrame):
    issues = []

    # ---------------------------------------------------------
    # 1. Duplicate rows
    # ---------------------------------------------------------
    if "symbol" not in df.columns:
        add_issue(issues, "CRITICAL", "Global file missing 'symbol' column.")
        return issues

    if df.duplicated(subset=["symbol", "date"]).any():
        add_issue(issues, "CRITICAL", "Duplicate (symbol, date) rows detected.")

    # ---------------------------------------------------------
    # 2. Symbol coverage / short histories
    # ---------------------------------------------------------
    counts = df["symbol"].value_counts()
    uneven = counts[counts < counts.median() * 0.5]
    if len(uneven) > 0:
        add_issue(
            issues, "WARNING",
            f"Tickers with unusually short histories: {list(uneven.index)}"
        )

    # ---------------------------------------------------------
    # 3. Date alignment
    # ---------------------------------------------------------
    ranges = df.groupby("symbol")["date"].agg(["min", "max"])
    max_end = ranges["max"].max()
    misaligned = ranges[ranges["max"] < max_end - pd.Timedelta(days=7)]
    if len(misaligned) > 0:
        add_issue(
            issues, "WARNING",
            f"Tickers ending more than 7 days early: {misaligned.index.tolist()}"
        )

    # ---------------------------------------------------------
    # 4. Global NaN floods
    # ---------------------------------------------------------
    nan_frac = df.isna().mean()
    critical_nan = nan_frac[nan_frac > 0.3]
    for col, frac in critical_nan.items():
        add_issue(issues, "WARNING", f"Global file: {col} is {frac*100:.1f}% NaN.")

    # ---------------------------------------------------------
    # 5. Cross-ticker RV/IV sanity
    # ---------------------------------------------------------
    if "RV_14" in df.columns:
        rv_by_symbol = df.groupby("symbol")["RV_14"].mean()
        if rv_by_symbol.std() > rv_by_symbol.mean() * 0.7:
            add_issue(issues, "WARNING", "Large cross-ticker divergence in RV_14.")

    if "IV_14" in df.columns:
        iv_by_symbol = df.groupby("symbol")["IV_14"].mean()
        if iv_by_symbol.std() > iv_by_symbol.mean() * 0.7:
            add_issue(issues, "WARNING", "Large cross-ticker divergence in IV_14.")

    # ---------------------------------------------------------
    # 6. Target leakage
    # ---------------------------------------------------------
    if "rv_14_forward" in df.columns:
        forward_by_symbol = df.groupby("symbol")["rv_14_forward"].mean()
        if forward_by_symbol.std() < 1e-6:
            add_issue(issues, "CRITICAL", "rv_14_forward identical across symbols (LEAK!).")

    return issues


# ============================================================
# MAIN PROCESS
# ============================================================

if __name__ == "__main__":

    # ---------------------------------------------------------
    # 1. PER-TICKER VALIDATION
    # ---------------------------------------------------------
    global_summary = {}

    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(".parquet"):
            continue

        symbol = fname.replace("_raw.parquet", "")
        df = pd.read_parquet(os.path.join(RAW_DIR, fname))

        print(f"\n=== Validating {symbol} ===")
        issues = validate_ticker(df, symbol)
        global_summary[symbol] = issues

        # Write individual reports
        txt = os.path.join(OUT_DIR, f"validation_report_{symbol}.txt")
        jsn = os.path.join(OUT_DIR, f"validation_report_{symbol}.json")

        with open(txt, "w", encoding="utf-8") as f:
            if not issues:
                f.write("CLEAN ✓\n")
            else:
                for item in issues:
                    f.write(f"[{item['level']}] {item['message']}\n")

        with open(jsn, "w", encoding="utf-8") as f:
            json.dump({"symbol": symbol, "issues": issues}, f, indent=2)

        print(f" → Reports written: {txt}, {jsn}")

    # ---------------------------------------------------------
    # 2. GLOBAL VALIDATION
    # ---------------------------------------------------------
    print("\n=== Validating GLOBAL all_tickers_raw.parquet ===")

    if not os.path.exists(GLOBAL_PATH):
        print("❌ Global file missing — skipping global validation.")
        global_issues = []
    else:
        df_global = pd.read_parquet(GLOBAL_PATH)
        global_issues = validate_global(df_global)

        txt = os.path.join(OUT_DIR, "global_validation.txt")
        jsn = os.path.join(OUT_DIR, "global_validation.json")

        with open(txt, "w", encoding="utf-8") as f:
            if not global_issues:
                f.write("GLOBAL CLEAN ✓\n")
            else:
                for item in global_issues:
                    f.write(f"[{item['level']}] {item['message']}\n")

        with open(jsn, "w", encoding="utf-8") as f:
            json.dump({"issues": global_issues}, f, indent=2)

        print(f" → Global reports written: {txt}, {jsn}")

    # ---------------------------------------------------------
    # 3A. HEALTH SCORE CALCULATION
    # ---------------------------------------------------------
    health_scores = {}
    for symbol, issues in global_summary.items():
        if not issues:
            health_scores[symbol] = 100
            continue

        n_crit = sum(1 for x in issues if x["level"] == "CRITICAL")
        n_warn = sum(1 for x in issues if x["level"] == "WARNING")

        score = 100 - 40*n_crit - 10*n_warn
        health_scores[symbol] = max(0, min(100, score))

    # ---------------------------------------------------------
    # 3B. CONSOLIDATED SUMMARY
    # ---------------------------------------------------------
    consolidated = {}
    for symbol, issues in global_summary.items():
        if not issues:
            consolidated[symbol] = ["CLEAN ✓"]
        else:
            consolidated[symbol] = [f"{x['level']}: {x['message']}" for x in issues]

    # Human-readable TXT
    summary_txt = os.path.join(OUT_DIR, "consolidated_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        for symbol, msgs in consolidated.items():
            score = health_scores[symbol]
            if msgs == ["CLEAN ✓"]:
                f.write(f"{symbol}: CLEAN ✓ | Health: {score}\n")
            else:
                f.write(f"{symbol}: {'; '.join(msgs)} | Health: {score}\n")

    # Machine-readable JSON
    summary_json = os.path.join(OUT_DIR, "consolidated_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tickers": consolidated,
                "health_scores": health_scores,
                "global_issues": global_issues
            },
            f,
            indent=2
        )

    print(f"\n → Consolidated summaries written: {summary_txt}, {summary_json}")
    print("\n🎉 Validation complete.")
