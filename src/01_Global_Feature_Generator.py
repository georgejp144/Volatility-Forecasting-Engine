# ============================================================
# Volatility Forecasting Engine — Raw Feature Builder
#
# PURPOSE:
#   Build deterministic, per-symbol daily feature sets including:
#       • Prices, returns, realised volatility
#       • Macro, FX, commodities, yields
#       • Technical indicators & microstructure
#       • Event-calendar overlays
#       • Target creation (14-day forward RV)
#
# USED FOR:
#   • Training 14-day RV forecasting models
#   • Walk-forward validation
#   • Signal generation for trading
#
# INPUTS:
#   symbols.csv                → list of tickers
#   events_manual.csv          → macro event specification
#
# OUTPUTS:
#   batch_vol_runs/global_features/raw_per_ticker/<SYMBOL>_raw.parquet
#   batch_vol_runs/global_features/all_tickers_raw.parquet
#
# RUN:
#   01_Global_Feature_Generator.py
#
# NOTES:
#   - Built to be deterministic per ticker on re-run
#   - Fully missing-data safe
#   - Safe around CPI window arithmetic
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
import requests
import yfinance as yf

from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, ADXIndicator

from scipy.stats import norm, skew, kurtosis


# ============================================================
# GLOBAL CONFIG
# ============================================================

YEARS_HISTORY = 7
RV_WINDOW = 14
TRADING_DAYS = 252
TARGET_DAYS_IV = 14

RAW_OUT_DIR = "batch_vol_runs/global_features/raw_per_ticker"
ALL_RAW_OUT_PATH = "batch_vol_runs/global_features/all_tickers_raw.parquet"
os.makedirs(RAW_OUT_DIR, exist_ok=True)

EVENTS_CSV_PATH = "events_manual.csv"

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")


# ============================================================
# EVENT CALENDAR CONSTRUCTION (CPI FIX APPLIED)
# ============================================================

def build_event_feature_calendar(events_csv_path: str) -> pd.DataFrame:
    """
    Construct a daily event feature calendar, mapping:
        • CPI pre/post windows
        • Macro and vol event pressure
        • Next-event importance
        • Stable days_to_next_cpi
    """

    base_cols = [
        "date",
        "days_to_next_cpi",
        "cpi_in_pre_window",
        "cpi_in_post_window",
        "macro_event_pressure",
        "vol_event_pressure",
        "next_event_importance",
    ]

    # If csv missing → return empty frame
    if not os.path.exists(events_csv_path):
        print("⚠️ Event CSV not found — disabling event features.")
        return pd.DataFrame(columns=base_cols)

    # Excel-safe encoding
    events = pd.read_csv(events_csv_path, encoding="cp1252")

    # Dates & numeric coercion
    events["event_date"] = (
        pd.to_datetime(events["event_date"], dayfirst=True, errors="coerce")
        .dt.normalize()
    )
    events["importance"] = pd.to_numeric(events["importance"], errors="coerce").fillna(0)
    events["window_pre"] = pd.to_numeric(events["window_pre"], errors="coerce").fillna(0)
    events["window_post"] = pd.to_numeric(events["window_post"], errors="coerce").fillna(0)

    events = events.dropna(subset=["event_date"])
    if events.empty:
        print("⚠️ Event CSV has no valid dates.")
        return pd.DataFrame(columns=base_cols)

    # Daily frame for full event span
    max_pre = events["window_pre"].max()
    max_post = events["window_post"].max()

    start_date = events["event_date"].min() - pd.Timedelta(days=max_pre)
    end_date = events["event_date"].max() + pd.Timedelta(days=max_post)

    cal = pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="D")})
    cal["days_to_next_cpi"] = np.nan
    cal["cpi_in_pre_window"] = 0.0
    cal["cpi_in_post_window"] = 0.0
    cal["macro_event_pressure"] = 0.0
    cal["vol_event_pressure"] = 0.0
    cal["next_event_importance"] = np.nan

    macro_types = {"CPI", "PPI", "NFP", "FOMC"}
    vol_types = {"OPEX", "VIX_EXPIRY", "ETF_REBAL"}

    # ------------------------------------------------------------
    # Pre & Post windows
    # ------------------------------------------------------------
    for _, ev in events.iterrows():
        ev_date = ev["event_date"]
        ev_type = str(ev["event_type"])
        imp = float(ev["importance"])
        pre = int(ev["window_pre"])
        post = int(ev["window_post"])

        # Pre-window
        if pre > 0:
            mask = (cal["date"] >= ev_date - pd.Timedelta(days=pre)) & (cal["date"] < ev_date)
            if ev_type == "CPI":
                cal.loc[mask, "cpi_in_pre_window"] = 1.0
            if ev_type in macro_types:
                cal.loc[mask, "macro_event_pressure"] += imp
            if ev_type in vol_types:
                cal.loc[mask, "vol_event_pressure"] += imp

        # Post-window (includes event day)
        mask = (cal["date"] >= ev_date) & (cal["date"] <= ev_date + pd.Timedelta(days=post))
        if ev_type == "CPI":
            cal.loc[mask, "cpi_in_post_window"] = 1.0
        if ev_type in macro_types:
            cal.loc[mask, "macro_event_pressure"] += imp
        if ev_type in vol_types:
            cal.loc[mask, "vol_event_pressure"] += imp

    # ------------------------------------------------------------
    # Next-event importance
    # ------------------------------------------------------------
    ev_sorted = events.sort_values("event_date")
    ev_dates = ev_sorted["event_date"].values.astype("datetime64[ns]")
    ev_imps = ev_sorted["importance"].values
    cal_dates = cal["date"].values.astype("datetime64[ns]")

    idx = np.searchsorted(ev_dates, cal_dates, side="left")
    next_imp = np.full(len(cal), np.nan)
    valid = idx < len(ev_dates)
    next_imp[valid] = ev_imps[idx[valid]]
    cal["next_event_importance"] = next_imp

    # ------------------------------------------------------------
    # CPI FIX — stable timedelta arithmetic
    # ------------------------------------------------------------
    cpi_df = events[events["event_type"] == "CPI"].sort_values("event_date")
    if not cpi_df.empty:
        cpi_dates = cpi_df["event_date"].values.astype("datetime64[ns]")

        cidx = np.searchsorted(cpi_dates, cal_dates, side="left")
        has = cidx < len(cpi_dates)

        next_cpi_vec = np.full(len(cal), np.datetime64("NaT"), dtype="datetime64[ns]")
        next_cpi_vec[has] = cpi_dates[cidx[has]]

        delta = (next_cpi_vec - cal_dates).astype("timedelta64[D]").astype(float)
        delta[~has] = np.nan
        cal["days_to_next_cpi"] = delta

    return cal


EVENT_FEATURES_DAILY = build_event_feature_calendar(EVENTS_CSV_PATH)


# ============================================================
# UTILITY HELPERS
# ============================================================

def normalise_dates(df):
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df


def flatten_columns(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def rolling_shannon_entropy(series, window=14, bins=10):
    def _entropy(x):
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.nan
        hist, _ = np.histogram(x, bins=bins)
        p = hist.astype(float)
        p = p[p > 0]
        if p.size == 0:
            return np.nan
        p /= p.sum()
        return -np.sum(p * np.log(p))

    return series.rolling(window).apply(_entropy, raw=False)


def rolling_autocorr(series, window, lag=1):
    def _acf(x):
        x = np.asarray(x)
        if x.size <= lag:
            return np.nan
        xm = x.mean()
        num = np.sum((x[:-lag] - xm) * (x[lag:] - xm))
        den = np.sum((x - xm) ** 2)
        return num / den if den != 0 else np.nan

    return series.rolling(window).apply(_acf, raw=True)


# ============================================================
# DATA FETCH — YAHOO / FRED
# ============================================================

def fetch_yahoo_ohlcv(symbol, start_naive, end_naive):
    """
    Download daily OHLCV data and forward/backfill to full daily frequency.
    """
    df = yf.download(
        symbol,
        start=start_naive,
        end=end_naive + timedelta(days=1),
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No Yahoo OHLCV for {symbol}")

    df = flatten_columns(df)
    df = df.reset_index().rename(columns={"Date": "date"})
    df = normalise_dates(df)

    df = df.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
    )[["date", "open", "high", "low", "close", "volume"]]

    df = df.set_index("date").asfreq("D").ffill().bfill().reset_index()
    return df


def fetch_fred_series(series_id, start_naive, end_naive):
    """
    Fetch daily macro series from FRED API, forward/backfill, normalised to daily.
    """
    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY missing")

    url = (
        "https://api.stlouisfed.org/fred/series/observations?"
        f"series_id={series_id}&api_key={FRED_API_KEY}&file_type=json"
        f"&observation_start={start_naive.date()}&observation_end={end_naive.date()}"
    )

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])

    if not obs:
        raise RuntimeError(f"No FRED observations for {series_id}")

    df = pd.DataFrame(obs)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df[["date", "value"]]
    df = df.set_index("date").asfreq("D").ffill().bfill().reset_index()
    return df


def fetch_yahoo_close(ticker, col_name, start_naive, end_naive):
    """
    Fetch daily close series for a single ticker (Close or Adj Close).
    """
    df = yf.download(
        ticker,
        start=start_naive,
        end=end_naive + timedelta(days=1),
        progress=False,
        auto_adjust=False,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No Yahoo for {ticker}")

    df = flatten_columns(df)
    df = df.reset_index().rename(columns={"Date": "date"})
    df = normalise_dates(df)

    close_col = "Close" if "Close" in df.columns else "Adj Close"
    df = df[["date", close_col]].rename(columns={close_col: col_name})

    df = df.set_index("date").asfreq("D").ffill().bfill().reset_index()
    return df


# ============================================================
# BLACK–SCHOLES + ATM IV (14-Day)
# ============================================================

def black_scholes_call_price(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return np.nan
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_call_iv(S, K, T, r, price):
    intrinsic = max(S - K * np.exp(-r * T), 0)
    if price < intrinsic:
        return np.nan

    def f(sig):
        return black_scholes_call_price(S, K, T, r, sig) - price

    try:
        from scipy.optimize import brentq
        return brentq(f, 1e-6, 5.0)
    except:
        return np.nan


def get_latest_atm_iv_14d(df_prices, symbol):
    ticker = yf.Ticker(symbol)

    try:
        opt_dates = ticker.options
    except:
        return np.nan

    if not opt_dates:
        return np.nan

    today = datetime.now(timezone.utc)

    # Choose expiry closest to 14 days
    expiry = min(
        opt_dates,
        key=lambda d: abs(
            (datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc) - today).days
            - TARGET_DAYS_IV
        ),
    )

    try:
        chain = ticker.option_chain(expiry).calls
    except:
        return np.nan

    if chain.empty:
        return np.nan

    S = float(df_prices["close"].iloc[-1])
    chain["diff"] = (chain["strike"] - S).abs()

    # Best price logic
    def best_px(r):
        bid, ask, last = r.get("bid", 0), r.get("ask", 0), r.get("lastPrice", 0)
        if bid > 0 and ask > 0:
            return 0.5 * (bid + ask)
        if last > 0:
            return last
        if bid > 0:
            return bid
        if ask > 0:
            return ask
        return np.nan

    T = TARGET_DAYS_IV / 365.0

    for _, r in chain.sort_values("diff").iterrows():
        px = best_px(r)
        if px and px > 0:
            iv = black_scholes_call_iv(S, float(r["strike"]), T, 0.0, px)
            if np.isfinite(iv) and iv > 0:
                return iv * 100.0

    return np.nan


# ============================================================
# MAIN FEATURE BUILDER (PER SYMBOL)
# ============================================================

def build_raw_features(symbol):
    """
    Build per-symbol feature set:
        • OHLCV backbone
        • Macro, FX, commodities
        • Yields, implied vol
        • Realised vol ladder (HAR + extended)
        • Technical + microstructure
        • Event calendar overlays
        • Forward 14-day RV target
    """

    # Date range
    end_utc = datetime.now(timezone.utc)
    start_utc = end_utc - timedelta(days=YEARS_HISTORY * 365)

    start_naive = start_utc.replace(tzinfo=None)
    end_naive = end_utc.replace(tzinfo=None)

    # --------------------------------------------------------
    # OHLCV backbone
    # --------------------------------------------------------
    df = fetch_yahoo_ohlcv(symbol, start_naive, end_naive)

    # --------------------------------------------------------
    # Macro — Yahoo
    # --------------------------------------------------------
    yahoo_sets = {
        "vix_close": "^VIX",
        "vxn_close": "^VXN",
        "move_index": "^MOVE",
        "hyg_close": "HYG",
        "fx_gbpusd": "GBPUSD=X",
        "fx_eurusd": "EURUSD=X",
        "fx_jpyusd": "JPY=X",
        "gold": "GC=F",
        "oil": "CL=F",
        "copper": "HG=F",
        "silver": "SI=F",
        "dxy_index": "DX-Y.NYB",
    }

    for col, tick in yahoo_sets.items():
        try:
            sub = fetch_yahoo_close(tick, col, start_naive, end_naive)
            df = df.merge(sub, on="date", how="left")
        except Exception as e:
            print(f"⚠️ Yahoo macro failed {tick}: {e}")
            df[col] = np.nan

    df["vix_vxn_spread"] = df["vix_close"] - df["vxn_close"]

    # --------------------------------------------------------
    # Macro — FRED
    # --------------------------------------------------------
    fred_sets = {
        "us10y_yield": "DGS10",
        "t3m_yield": "DGS3MO",
        "us5y_yield": "DGS5",
        "us30y_yield": "DGS30",
    }

    for col, sid in fred_sets.items():
        try:
            fred = fetch_fred_series(sid, start_naive, end_naive)
            df = df.merge(fred.rename(columns={"value": col}), on="date", how="left")
        except Exception as e:
            print(f"⚠️ FRED fetch failed {sid}: {e}")
            df[col] = np.nan

    # --------------------------------------------------------
    # Implied Vol (ATM 14-Day)
    # --------------------------------------------------------
    iv = get_latest_atm_iv_14d(df, symbol)
    df["IV_14"] = iv
    df["IV_14"] = df["IV_14"].ffill().bfill()

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = np.log(df["close"]).diff()

    # RV
    df["RV_14_raw"] = (
        df["ret"].rolling(RV_WINDOW).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
        * np.sqrt(TRADING_DAYS)
    )
    df["RV_14"] = df["RV_14_raw"] * 100.0

    # HAR ladder
    df["rv_1"] = df["RV_14_raw"].shift(1)
    df["rv_5"] = df["ret"].rolling(5).std() * np.sqrt(TRADING_DAYS)
    df["rv_22"] = df["ret"].rolling(22).std() * np.sqrt(TRADING_DAYS)
    df["rv_63"] = df["ret"].rolling(63).std() * np.sqrt(TRADING_DAYS)
    df["rv_5_22_ratio"] = df["rv_5"] / df["rv_22"]

    # Extended RV ladder
    df["RV_2"] = df["ret"].rolling(2).std() * np.sqrt(TRADING_DAYS)
    df["RV_3"] = df["ret"].rolling(3).std() * np.sqrt(TRADING_DAYS)
    df["RV_7"] = df["ret"].rolling(7).std() * np.sqrt(TRADING_DAYS)
    df["RV_21"] = df["ret"].rolling(21).std() * np.sqrt(TRADING_DAYS)
    df["RV_42"] = df["ret"].rolling(42).std() * np.sqrt(TRADING_DAYS)
    df["RV_126"] = df["ret"].rolling(126).std() * np.sqrt(TRADING_DAYS)
    df["RV_252"] = df["ret"].rolling(252).std() * np.sqrt(TRADING_DAYS)

    df["rv_7_21_ratio"] = df["RV_7"] / df["RV_21"]
    df["rv_21_63_ratio"] = df["RV_21"] / df["rv_63"]
    df["rv_14_63_ratio"] = df["RV_14_raw"] / df["rv_63"]
    df["rv_7_126_ratio"] = df["RV_7"] / df["RV_126"]

    # --------------------------------------------------------
    # Compression + Jumps
    # --------------------------------------------------------
    df["range_pct"] = (df["high"] - df["low"]) / df["close"] * 100.0
    df["prev_close"] = df["close"].shift(1)
    df["gap_pct"] = (df["open"] - df["prev_close"]) / df["prev_close"] * 100.0

    bb = BollingerBands(df["close"], 20, 2)
    df["bb_bandwidth"] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    df["bb_pctB"] = (df["close"] - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband()
    )

    atr = AverageTrueRange(df["high"], df["low"], df["close"], 14)
    df["atr_14"] = atr.average_true_range()
    df["atr_pct"] = df["atr_14"] / df["close"] * 100.0

    # Parkinson, Garman-Klass, Rogers-Satchell
    hl_log = np.log(df["high"] / df["low"])
    df["parkinson_vol_14"] = (
        np.sqrt(
            (1 / (4 * RV_WINDOW * np.log(2))) * (hl_log**2).rolling(RV_WINDOW).sum()
        )
        * np.sqrt(TRADING_DAYS)
    )

    log_hl = np.log(df["high"] / df["low"])
    log_co = np.log(df["close"] / df["open"].replace(0, np.nan))
    df["gk_var_daily"] = 0.5 * (log_hl**2) - (2 * np.log(2) - 1) * (log_co**2)
    df["gk_vol_14"] = np.sqrt(df["gk_var_daily"].rolling(RV_WINDOW).mean() * TRADING_DAYS)

    log_ho = np.log(df["high"] / df["open"].replace(0, np.nan))
    log_lo = np.log(df["low"] / df["open"].replace(0, np.nan))
    log_co2 = np.log(df["close"] / df["open"].replace(0, np.nan))
    df["rs_var_daily"] = log_ho * (log_ho - log_co2) + log_lo * (log_lo - log_co2)
    df["rs_vol_14"] = np.sqrt(df["rs_var_daily"].rolling(RV_WINDOW).mean() * TRADING_DAYS)

    df["range_pct_vol_14"] = df["range_pct"].rolling(RV_WINDOW).std()
    df["ret_entropy_14"] = rolling_shannon_entropy(df["ret"], window=RV_WINDOW, bins=10)

    # Skew / kurtosis
    df["realised_skew_14"] = df["ret"].rolling(RV_WINDOW).apply(lambda x: skew(x, bias=False))
    df["realised_kurtosis_14"] = df["ret"].rolling(RV_WINDOW).apply(
        lambda x: kurtosis(x, fisher=False, bias=False)
    )

    # Jumps
    ret_std_14 = df["ret"].rolling(RV_WINDOW).std()
    df["jump_flag"] = (np.abs(df["ret"]) > 2.5 * ret_std_14).astype(float)

    # --------------------------------------------------------
    # Vol-of-vol
    # --------------------------------------------------------
    df["vol_of_vol_14"] = df["RV_14_raw"].rolling(RV_WINDOW).std()
    df["vol_of_vol_30"] = df["RV_14_raw"].rolling(30).std()
    df["vol_of_vol_ratio_14"] = df["vol_of_vol_14"] / df["RV_14_raw"]

    df["RV_14_smooth_ema10"] = df["RV_14_raw"].ewm(span=10, adjust=False).mean()
    df["RV_14_deviation_from_ema10"] = df["RV_14_raw"] - df["RV_14_smooth_ema10"]

    # --------------------------------------------------------
    # Macro transforms
    # --------------------------------------------------------
    df["vix_over_rv14"] = df["vix_close"] / df["RV_14"]
    df["vix_minus_rv14"] = df["vix_close"] - df["RV_14"]
    df["move_minus_vix"] = df["move_index"] - df["vix_close"]
    df["vxn_over_vix"] = df["vxn_close"] / df["vix_close"]

    # FX vol index
    fx_cols = ["fx_gbpusd", "fx_eurusd", "fx_jpyusd"]
    fx_ret_cols = []
    for c in fx_cols:
        if c in df.columns:
            col_ret = f"{c}_ret"
            df[col_ret] = np.log(df[c]).diff()
            fx_ret_cols.append(col_ret)

    if fx_ret_cols:
        df["fx_vol_index_14"] = (
            df[fx_ret_cols].rolling(RV_WINDOW).std().mean(axis=1) * np.sqrt(TRADING_DAYS)
        )

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------
    df["rsi_14"] = RSIIndicator(df["close"], 14).rsi()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["sma_10_over_20"] = (df["sma_10"] > df["sma_20"]).astype(float)
    df["sma_20_over_50"] = (df["sma_20"] > df["sma_50"]).astype(float)

    macd = MACD(df["close"], 26, 12, 9)
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    adx = ADXIndicator(df["high"], df["low"], df["close"], 14)
    df["adx_14"] = adx.adx()

    # --------------------------------------------------------
    # Microstructure
    # --------------------------------------------------------
    df["volume_mean_20"] = df["volume"].rolling(20).mean()
    df["volume_std_20"] = df["volume"].rolling(20).std()
    df["volume_zscore_20"] = (df["volume"] - df["volume_mean_20"]) / df["volume_std_20"]
    df["volume_volatility_20"] = df["volume"].rolling(20).std()

    df["price_volume_corr_20"] = df["close"].rolling(20).corr(df["volume"])

    price_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["signed_volume"] = ((df["close"] - df["open"]) / price_range) * df["volume"]

    df["ret_autocorr_1_21"] = rolling_autocorr(df["ret"], window=21, lag=1)

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_22"] = df["close"].pct_change(22)

    # --------------------------------------------------------
    # Event-based features
    # --------------------------------------------------------
    if not EVENT_FEATURES_DAILY.empty:
        df = df.merge(EVENT_FEATURES_DAILY, on="date", how="left")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    else:
        df["days_to_next_cpi"] = np.nan
        df["cpi_in_pre_window"] = np.nan
        df["cpi_in_post_window"] = np.nan
        df["macro_event_pressure"] = np.nan
        df["vol_event_pressure"] = np.nan
        df["next_event_importance"] = np.nan

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------
    df["rv_14_forward"] = df["RV_14_raw"].shift(-RV_WINDOW)
    df["symbol"] = symbol

    # Drop warmup rows
    df = df.iloc[100:]

    # Per-symbol dedupe protection
    df = df.drop_duplicates(subset=["symbol", "date"], keep="last")
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    symbols = pd.read_csv("symbols.csv").iloc[:, 0].astype(str).str.strip().tolist()

    all_frames = []

    print(f"📡 Loaded {len(symbols)} symbols")

    for sym in symbols:
        print(f"\n=== Building features for {sym} ===")

        try:
            df_raw = build_raw_features(sym)

            # Per-symbol safety
            df_raw = df_raw.drop_duplicates(subset=["symbol", "date"], keep="last")
            df_raw = df_raw.sort_values(["symbol", "date"]).reset_index(drop=True)

            out_path = f"{RAW_OUT_DIR}/{sym}_raw.parquet"
            df_raw.to_parquet(out_path, index=False)

            print(f"🧩 Saved → {out_path} ({df_raw.shape[0]} rows, {df_raw.shape[1]} cols)")
            all_frames.append(df_raw)

        except Exception as e:
            print(f"❌ Failed for {sym}: {e}")

    # --------------------------------------------------------
    # Global combine & dedupe
    # --------------------------------------------------------
    if all_frames:
        all_df = pd.concat(all_frames, ignore_index=True)

        all_df = all_df.drop_duplicates(subset=["symbol", "date"], keep="last")
        all_df = all_df.sort_values(["symbol", "date"]).reset_index(drop=True)

        all_df.to_parquet(ALL_RAW_OUT_PATH, index=False)

        print(f"\n📦 Saved global dataset → {ALL_RAW_OUT_PATH}")
        print(f"   Total rows: {all_df.shape[0]} | Columns: {all_df.shape[1]}")

    else:
        print("⚠️ No frames produced — nothing saved.")
