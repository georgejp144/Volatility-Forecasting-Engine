# ============================================================
# signal_engine.py
# ============================================================
#
# PURPOSE
#   Generate trading signals from the model’s daily forecast summary.
#
# USED FOR
#   - Identify Long Gamma entry opportunities
#   - Identify Short Vega entry opportunities
#   - Detect exit conditions for existing Long Gamma trades
#   - Provide regime context and diversification indicators
#   - Rank opportunities by IV edge
#
# INPUTS
#   batch_vol_runs/summary_with_events.csv
#       Contains forecast outputs, IV–RV spreads, regime scores,
#       and event-window status for each symbol.
#
# OUTPUTS
#   batch_vol_runs/trade_signals_today.csv
#       A compact, execution-ready table containing:
#       - Trade signals (entry/exit)
#       - Regime classification
#       - Diversification tags
#       - Ranking by absolute IV edge
#       - Footer summary counts
#
# RESPONSIBILITIES
#   • Generate Long Gamma entry signals
#   • Generate Short Vega entry signals
#   • Generate Long Gamma exit signals
#   • Classify volatility regime
#   • Produce diversification tags
#   • Rank opportunities by IV mispricing
#   • Save minimal CSV for downstream execution
#
# NOTES
#   - No trading logic or position sizing included
#   - This module produces signals only
#   - Inputs come from batch runner (LSTM + XGB forecasts + events)
#
# ============================================================

import os
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

INPUT_SUMMARY = "batch_vol_runs/summary_with_events.csv"
OUTPUT_TRADE = "batch_vol_runs/trade_signals_today.csv"


# ============================================================
# LOAD SUMMARY
# ============================================================

def load_summary(path=INPUT_SUMMARY):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Could not find: {path}")
    df = pd.read_csv(path)
    return df


# ============================================================
# SIGNAL LOGIC FUNCTIONS
# ============================================================

def compute_long_gamma_entry(row):
    """
    Long Gamma Entry Conditions:
        1) IV < Forecast p10 (undervalued vol)
        2) Spread < 6 (narrow forecast distribution)
        3) Forecast RV well above current RV
        4) Compression environment
        5) Model reliability strong
        6) No macro window active
    """
    try:
        cond1 = row["IV_14"] < row["Ensemble_Forecast_RV_14_p10"]
        cond2 = row["Ensemble_Forecast_RV_14_Spread"] < 6
        cond3 = (row["Ensemble_Forecast_RV_14"] - row["RV_14"]) > 1.0
        cond4 = (row.get("bb_bandwidth_pctile", 999) < 30) and (row.get("atr_pct_pctile", 999) < 35)
        cond5 = row["Model_Reliability_Score"] > 60
        cond6 = not row["Macro_Window_Live"]

        return int(cond1 and cond2 and cond3 and cond4 and cond5 and cond6)
    except:
        return 0


def compute_short_vega_entry(row):
    """
    Short Vega Entry Conditions:
        1) IV > Forecast p90 (rich vol)
        2) Spread < 6
        3) Model reliability good
        4) No event windows active
    """
    try:
        cond1 = row["IV_14"] > row["Ensemble_Forecast_RV_14_p90"]
        cond2 = row["Ensemble_Forecast_RV_14_Spread"] < 6
        cond3 = row["Model_Reliability_Score"] > 65
        cond4 = (not row["Macro_Window_Live"]) and (not row["Vol_Window_Live"])

        return int(cond1 and cond2 and cond3 and cond4)
    except:
        return 0


def compute_long_gamma_exit(row):
    """
    Exit Long Gamma Conditions:
        Trigger exit if ANY of:
            - Forecast RV < Current RV
            - IV > Ensemble Forecast
            - Spread > 10
            - Macro event imminent (<= 3 days)
    """
    try:
        cond1 = row["Ensemble_Forecast_RV_14"] < row["RV_14"]
        cond2 = row["IV_14"] > row["Ensemble_Forecast_RV_14"]
        cond3 = row["Ensemble_Forecast_RV_14_Spread"] > 10
        cond4 = row["Days_To_Next_Macro_Event"] <= 3

        return int(cond1 or cond2 or cond3 or cond4)
    except:
        return 0


def classify_regime(row):
    """
    Volatility Regime:
        - >=70 → STABLE_REGIME
        - >=40 → MIXED_REGIME
        - else → CHAOTIC_REGIME
    """
    mrs = row["Model_Reliability_Score"]
    if mrs >= 70:
        return "STABLE_REGIME"
    elif mrs >= 40:
        return "MIXED_REGIME"
    else:
        return "CHAOTIC_REGIME"


def diversification_tag(row):
    """
    Diversification Tags:
        Uses pressure comparison to determine primary driver.
    """
    m = row["Macro_Event_Pressure"]
    v = row["Vol_Event_Pressure"]

    if m > v:
        return "Primary pick — Macro Regime"
    elif v > m:
        return "Primary pick — Vol Regime"
    else:
        return "Secondary pick — Neutral"


# ============================================================
# MAIN
# ============================================================

def main():

    print("📥 Loading summary_with_events.csv ...")
    df = load_summary(INPUT_SUMMARY)

    # ============================================================
    # CALCULATE SIGNALS
    # ============================================================

    df["LongGamma_Entry"] = df.apply(compute_long_gamma_entry, axis=1)
    df["ShortVega_Entry"] = df.apply(compute_short_vega_entry, axis=1)
    df["End_LongGamma_Condition"] = df.apply(compute_long_gamma_exit, axis=1)

    # ============================================================
    # UTILITY METRICS
    # ============================================================

    if "Ensemble_vs_IV" in df.columns:
        df["Abs_Ensemble_vs_IV"] = df["Ensemble_vs_IV"].abs()
    else:
        print("⚠️ Ensemble_vs_IV missing — creating fallback using IV_14 - Ensemble_Forecast")
        df["Abs_Ensemble_vs_IV"] = (df["IV_14"] - df["Ensemble_Forecast_RV_14"]).abs()

    df["Volatility_Regime"] = df.apply(classify_regime, axis=1)
    df["Diversification_Tag"] = df.apply(diversification_tag, axis=1)

    # ============================================================
    # FILTER ACTIVE SIGNALS
    # ============================================================

    signal_mask = (
        (df["LongGamma_Entry"] == 1) |
        (df["ShortVega_Entry"] == 1) |
        (df["End_LongGamma_Condition"] == 1)
    )

    active = df[signal_mask].copy()

    essential_cols = [
        "symbol",
        "LongGamma_Entry",
        "ShortVega_Entry",
        "End_LongGamma_Condition",
        "Ensemble_vs_IV",
        "Abs_Ensemble_vs_IV",
        "Volatility_Regime",
        "Diversification_Tag",
    ]

    active = active[essential_cols].sort_values("Abs_Ensemble_vs_IV", ascending=False)

    # ============================================================
    # SUMMARY FOOTER
    # ============================================================

    lg_count = int((df["LongGamma_Entry"] == 1).sum())
    sv_count = int((df["ShortVega_Entry"] == 1).sum())
    exit_count = int((df["End_LongGamma_Condition"] == 1).sum())

    footer = {
        "symbol": "SUMMARY",
        "LongGamma_Entry": lg_count,
        "ShortVega_Entry": sv_count,
        "End_LongGamma_Condition": exit_count,
        "Ensemble_vs_IV": np.nan,
        "Abs_Ensemble_vs_IV": np.nan,
        "Volatility_Regime": "",
        "Diversification_Tag": "",
    }

    active = pd.concat([active, pd.DataFrame([{}]), pd.DataFrame([footer])], ignore_index=True)

    # ============================================================
    # SAVE TO DISK (Windows-safe overwrite)
    # ============================================================

    os.makedirs(os.path.dirname(OUTPUT_TRADE), exist_ok=True)

    try:
        if os.path.exists(OUTPUT_TRADE):
            os.remove(OUTPUT_TRADE)
    except PermissionError:
        print("❌ File is open in Excel — close Excel and rerun.")
        return

    active.to_csv(OUTPUT_TRADE, index=False)

    # ============================================================
    # LOG
    # ============================================================

    print(f"✅ trade_signals_today.csv written → {OUTPUT_TRADE}")
    print(f"   LongGamma entries: {lg_count}")
    print(f"   ShortVega entries: {sv_count}")
    print(f"   Exit signals: {exit_count}")


if __name__ == "__main__":
    main()
