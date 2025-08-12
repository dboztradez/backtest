#!/usr/bin/env python3
# backtest.py — EURUSD M5 DR/IDR + ORB hybrid
# Runs fully in GitHub Codespaces. Outputs CSVs + JSON.

import os, sys, json, math, argparse, time as pytime
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
from dateutil import tz
import requests

NY = tz.gettz("America/New_York")

# ========== Config defaults (you can override with CLI flags later) ==========
PAIR = "EURUSD"
RISK_PCT = 1.0               # 1% per trade
MAX_DAILY_LOSS = 5.0         # 5% daily loss cap
ADX_LEN = 14
ADX_MIN = 25.0
VOL_LOOKBACK = 50
STRONG_BODY_PCT = 60.0
DR_TRADES_CAP = 4
ORB_TRADES_CAP = 2
NEWS_EMBARGO_MIN = 15        # +/- minutes (optional; requires events list)
ONE_OPEN_AT_A_TIME = True

# ========== Utilities ==========
def to_et(ts: pd.Timestamp) -> pd.Timestamp:
    return ts.tz_convert(NY)

def hhmm(dt: pd.Timestamp) -> int:
    return dt.hour * 100 + dt.minute

def in_window(et_dt: pd.Timestamp, start_hm: int, end_hm: int) -> bool:
    """True if et_dt in [start, end) (NY time). Handles windows that cross midnight."""
    sH, sM = divmod(start_hm, 100)
    eH, eM = divmod(end_hm, 100)
    s = et_dt.replace(hour=sH, minute=sM, second=0, microsecond=0)
    e = et_dt.replace(hour=eH, minute=eM, second=0, microsecond=0)
    if end_hm <= start_hm:  # cross-midnight
        if et_dt < s:
            s -= timedelta(days=1)
        e += timedelta(days=1)
    return s <= et_dt < e

def weekday(dt: pd.Timestamp) -> int:
    return dt.weekday()  # Mon=0 .. Sun=6

# ========== Data adapters ==========
def fetch_alpha(from_date: str, to_date: str, interval="5min") -> pd.DataFrame:
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "FX_INTRADAY",
        "from_symbol": "EUR",
        "to_symbol": "USD",
        "interval": interval,
        "outputsize": "full",
        "apikey": key,
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    js = r.json()
    k = f"Time Series FX ({interval})"
    if k not in js:
        raise RuntimeError(f"AlphaVantage response missing data: {js}")
    rows = []
    for ts, ohlc in js[k].items():
        rows.append({
            "time": ts,
            "open": float(ohlc["1. open"]),
            "high": float(ohlc["2. high"]),
            "low":  float(ohlc["3. low"]),
            "close":float(ohlc["4. close"]),
            "volume": float(ohlc.get("5. volume", 0.0)),
        })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time")
    start = pd.to_datetime(from_date, utc=True)
    end = pd.to_datetime(to_date, utc=True) + pd.Timedelta(days=1)  # inclusive
    df = df[(df.time >= start) & (df.time < end)].reset_index(drop=True)
    return df

def fetch_twelve(from_date: str, to_date: str, interval="5min") -> pd.DataFrame:
    key = os.getenv("TWELVEDATA_API_KEY")
    if not key:
        raise RuntimeError("TWELVEDATA_API_KEY not set")
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": "EUR/USD",
        "interval": interval,
        "start_date": from_date,
        "end_date": to_date,
        "format": "JSON",
        "apikey": key,
        "timezone": "UTC",
        "dp": 6,
    }
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    js = r.json()
    if "values" not in js:
        raise RuntimeError(f"TwelveData error: {js}")
    rows = []
    for v in js["values"]:
        rows.append({
            "time": v["datetime"],
            "open": float(v["open"]),
            "high": float(v["high"]),
            "low":  float(v["low"]),
            "close":float(v["close"]),
            "volume": float(v.get("volume", 0.0)),
        })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)

def fetch_oanda(from_date: str, to_date: str, instrument="EUR_USD", granularity="M5") -> pd.DataFrame:
    key = os.getenv("OANDA_API_KEY")
    if not key:
        raise RuntimeError("OANDA_API_KEY not set (practice)")
    base = "https://api-fxpractice.oanda.com"
    url = f"{base}/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {key}"}
    params = {
        "price": "M",
        "granularity": granularity,
        "from": from_date + "T00:00:00Z",
        "to": to_date + "T23:59:59Z",
        "count": 5000,
    }
    r = requests.get(url, headers=headers, params=params, timeout=90)
    r.raise_for_status()
    js = r.json()
    rows = []
    for c in js.get("candles", []):
        if not c.get("complete"):
            continue
        rows.append({
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low":  float(c["mid"]["l"]),
            "close":float(c["mid"]["c"]),
            "volume": float(c.get("volume", 0.0)),
        })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)

# ========== Indicators ==========
def adx(df: pd.DataFrame, length=14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift()).abs()
    tr3 = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    pdi = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / atr).fillna(0)
    mdi = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=1/length, adjust=False).mean() / atr).fillna(0)
    dx = 100 * (abs(pdi - mdi) / (pdi + mdi)).replace([np.inf, -np.inf], 0).fillna(0)
    return dx.ewm(alpha=1/length, adjust=False).mean()

def vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"].replace(0, np.nan)
    return (pv.cumsum() / df["volume"].replace(0, np.nan).cumsum()).fillna(method="ffill")

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()

def ema_on_tf(close: pd.Series, index: pd.DatetimeIndex, tf_minutes: int, length: int) -> pd.Series:
    """M15 EMA200 sampled on M5 bars: resample to 15m, EMA(200), ffill."""
    df = pd.DataFrame({"close": close}, index=index)
    res = df["close"].resample(f"{tf_minutes}T").last()
    ema_tf = ema(res, length)
    ema_upsampled = ema_tf.reindex(index.union(ema_tf.index)).ffill().reindex(index).ffill()
    return ema_upsampled

# ========== Ranges ==========
def compute_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Accumulate DR1 (04:30–05:30), DR2/ORB (09:30–09:45), ORB (same window) per NY day."""
    df = df.copy()
    df["et"] = df["time"].dt.tz_convert(NY)
    # init
    dr1h = dr1l = np.nan
    dr2h = dr2l = np.nan
    orbh = orbl = np.nan
    on_dr1 = on_dr2 = on_orb = False

    out = {
        "DR1_high": [], "DR1_low": [],
        "DR2_high": [], "DR2_low": [],
        "ORB_high": [], "ORB_low": []
    }

    for i, row in df.iterrows():
        et = row["et"]

        # window flags
        in_dr1 = in_window(et, 430, 530)
        in_dr2 = in_window(et, 930, 945)
        in_orb = in_dr2  # same range window

        # start edges
        if in_dr1 and not on_dr1:
            on_dr1 = True; dr1h = row["high"]; dr1l = row["low"]
        if on_dr1 and in_dr1:
            dr1h = max(dr1h, row["high"]); dr1l = min(dr1l, row["low"])
        if on_dr1 and not in_dr1:
            on_dr1 = False

        if in_dr2 and not on_dr2:
            on_dr2 = True; dr2h = row["high"]; dr2l = row["low"]
        if on_dr2 and in_dr2:
            dr2h = max(dr2h, row["high"]); dr2l = min(dr2l, row["low"])
        if on_dr2 and not in_dr2:
            on_dr2 = False

        if in_orb and not on_orb:
            on_orb = True; orbh = row["high"]; orbl = row["low"]
        if on_orb and in_orb:
            orbh = max(orbh, row["high"]); orbl = min(orbl, row["low"])
        if on_orb and not in_orb:
            on_orb = False

        out["DR1_high"].append(dr1h); out["DR1_low"].append(dr1l)
        out["DR2_high"].append(dr2h); out["DR2_low"].append(dr2l)
        out["ORB_high"].append(orbh); out["ORB_low"].append(orbl)

    for k, v in out.items():
        df[k] = v

    return df

# ========== News embargo (optional) ==========
def load_events_csv(path: Optional[str]) -> List[pd.Timestamp]:
    """CSV with a 'time' column in UTC (ISO)."""
    if not path:
        return []
    ev = pd.read_csv(path)
    if "time" not in ev.columns:
        raise RuntimeError("events CSV must have a 'time' column (UTC ISO timestamps)")
    ts = pd.to_datetime(ev["time"], utc=True)
    return list(ts.sort_values())

def in_embargo(utc_ts: pd.Timestamp, events: List[pd.Timestamp], minutes=15) -> bool:
    if not events:
        return False
    lo = utc_ts - pd.Timedelta(minutes=minutes)
    hi = utc_ts + pd.Timedelta(minutes=minutes)
    # quick scan (events list is small typically)
    for e in events:
        if lo <= e <= hi:
            return True
    return False

# ========== Strategy backtest ==========
@dataclass
class Position:
    window: str   # "DRIDR1" / "DRIDR2" / "ORB"
    side: str     # "long" / "short"
    dir: int      # +1 / -1
    entry_time: pd.Timestamp
    entry: float
    sl: float
    tp: Optional[float]
    units: float

def run_backtest(df: pd.DataFrame,
                 risk_pct=RISK_PCT,
                 max_daily_loss_pct=MAX_DAILY_LOSS,
                 dr_cap=DR_TRADES_CAP,
                 orb_cap=ORB_TRADES_CAP,
                 events: Optional[List[pd.Timestamp]]=None,
                 use_embargo=True) -> Dict:

    df = df.copy()
    df["et"] = df["time"].dt.tz_convert(NY)
    df["date_et"] = df["et"].dt.date
    df = compute_ranges(df)
    # Indicators
    df["ADX"] = adx(df, ADX_LEN)
    df["VWAP"] = vwap(df)
    df["volZ"] = (df["volume"] - df["volume"].rolling(VOL_LOOKBACK).mean()) / df["volume"].rolling(VOL_LOOKBACK).std().replace(0, np.nan)
    df["volZ"] = df["volZ"].fillna(0)
    df["bodyPct"] = np.where(
        (df["high"] - df["low"]) > 0,
        100.0 * (abs(df["close"] - df["open"]) / (df["high"] - df["low"])),
        0.0
    )
    # M15 EMA200 bias on M5 bars
    df["EMA200_M15"] = ema_on_tf(df["close"], df["time"], 15, 200)

    start_bal = 10_000.0
    balance = start_bal
    equity = start_bal
    day_start_equity = None

    open_pos: Optional[Position] = None
    trades: List[Dict] = []
    equity_curve: List[Dict] = []

    # per-window trade counters (reset on window change)
    trades_this_window = 0
    active_window = "NONE"

    def pos_size(entry, stop) -> float:
        if stop is None or np.isnan(stop):
            return 0.0
        pip = 0.0001
        risk_cash = balance * (risk_pct / 100.0)
        sl_pips = max(1.0, abs(entry - stop) / pip)
        units = risk_cash / (sl_pips * 1.0)
        return max(0.0, units)

    for i, row in df.iterrows():
        et = row["et"]; wd = weekday(et)
        orb_trade = in_window(et, 945, 1600)
        dr_am     = in_window(et, 530, 930)
        dr_pm     = in_window(et, 1030, 430)  # next day

        new_active = "NONE"
        if wd in [0, 4]:  # Mon, Fri
            if dr_am: new_active = "DRIDR1"
            elif dr_pm: new_active = "DRIDR2"
        if wd in [1, 2, 3]:  # Tue-Thu
            if dr_am: new_active = "DRIDR1"
            elif orb_trade: new_active = "ORB"

        # reset counter on window change
        if new_active != active_window:
            trades_this_window = 0
            active_window = new_active

        # reset daily equity at NY day change
        if day_start_equity is None or (i > 0 and df.loc[i-1, "date_et"] != row["date_et"]):
            day_start_equity = equity

        # force exit on window close
        def window_is_open(name: str) -> bool:
            if name == "ORB": return orb_trade
            if name == "DRIDR1": return dr_am
            if name == "DRIDR2": return dr_pm
            return False

        if open_pos and not window_is_open(open_pos.window):
            exit_price = row["close"]
            pnl = (exit_price - open_pos.entry) * open_pos.dir * open_pos.units
            balance += pnl; equity = balance
            trades.append({
                **open_pos.__dict__,
                "exit_time": row["time"], "exit": exit_price, "pnl": pnl
            })
            open_pos = None

        # block new entries: daily loss or embargo
        if (equity - day_start_equity) / day_start_equity <= -(max_daily_loss_pct / 100.0):
            equity_curve.append({"time": row["time"], "equity": equity}); continue
        if use_embargo and in_embargo(row["time"], events or [], NEWS_EMBARGO_MIN):
            equity_curve.append({"time": row["time"], "equity": equity}); continue

        # common gates
        adx_ok = row["ADX"] >= ADX_MIN
        vwap_long_ok = row["close"] >= row["VWAP"]
        vwap_short_ok = row["close"] <= row["VWAP"]
        vol_spike = row["volZ"] >= 1.0
        strong_candle = row["bodyPct"] >= STRONG_BODY_PCT

        # ORB range
        orbh, orbl = row["ORB_high"], row["ORB_low"]
        # DR bounds (prefer DR2 when in DRIDR2)
        dr1h, dr1l, dr2h, dr2l = row["DR1_high"], row["DR1_low"], row["DR2_high"], row["DR2_low"]

        # FVG proxy (3-bar) — simple: up-gap/ down-gap
        if i >= 2:
            fvg_up = (df.loc[i-1, "low"] > df.loc[i-2, "high"])
            fvg_dn = (df.loc[i-1, "high"] < df.loc[i-2, "low"])
        else:
            fvg_up = fvg_dn = False

        # ========== ENTRY LOGIC ==========
        made_entry = False

        # ORB (Tue–Thu): breakout w/ EMA200 M15 bias + VWAP align + ADX + strong candle + vol spike
        if active_window == "ORB" and trades_this_window < ORB_TRADES_CAP and (not ONE_OPEN_AT_A_TIME or not open_pos):
            ema_bias_long = row["close"] >= row["EMA200_M15"]
            ema_bias_short = row["close"] <= row["EMA200_M15"]
            broke_up = (pd.notna(orbh) and row["close"] > orbh)
            broke_dn = (pd.notna(orbl) and row["close"] < orbl)

            # Allow FVG only after re-entry beyond OR boundary (re-entry proxy: close beyond, and prior bar pierced boundary)
            reentry_long = (pd.notna(orbh) and row["close"] > orbh and df.loc[i, "low"] < orbh)
            reentry_short = (pd.notna(orbl) and row["close"] < orbl and df.loc[i, "high"] > orbl)
            fvg_allowed_long = reentry_long and fvg_up
            fvg_allowed_short = reentry_short and fvg_dn

            if adx_ok and strong_candle and vol_spike:
                if broke_up and ema_bias_long and vwap_long_ok and fvg_allowed_long:
                    # simple ORB SL: 10 pips; TP none (let window close force exit)
                    sl = row["close"] - 0.0010
                    units = pos_size(row["close"], sl)
                    if units > 0:
                        open_pos = Position("ORB", "long", +1, row["time"], row["close"], sl, None, units)
                        trades_this_window += 1; made_entry = True
                elif broke_dn and ema_bias_short and vwap_short_ok and fvg_allowed_short:
                    sl = row["close"] + 0.0010
                    units = pos_size(row["close"], sl)
                    if units > 0:
                        open_pos = Position("ORB", "short", -1, row["time"], row["close"], sl, None, units)
                        trades_this_window += 1; made_entry = True

        # DR/IDR (Mon/Fri + Tue–Thu morning): bounce at bounds, immediate FVG allowed, SL 5p beyond, TP opposite bound
        if not made_entry and active_window in ("DRIDR1", "DRIDR2") and trades_this_window < DR_TRADES_CAP and (not ONE_OPEN_AT_A_TIME or not open_pos):
            # choose active DR bounds
            dr_low = dr2l if (active_window == "DRIDR2" and pd.notna(dr2l)) else dr1l
            dr_high = dr2h if (active_window == "DRIDR2" and pd.notna(dr2h)) else dr1h

            if pd.notna(dr_low) and pd.notna(dr_high) and adx_ok:
                # long bounce off lower range + VWAP align + simple reversal (close > open)
                if vwap_long_ok and (row["low"] <= dr_low + 0.00015) and (row["close"] > row["open"]):
                    sl = dr_low - 0.0005
                    tp = dr_high
                    units = pos_size(row["close"], sl)
                    if units > 0:
                        open_pos = Position(active_window, "long", +1, row["time"], row["close"], sl, tp, units)
                        trades_this_window += 1; made_entry = True
                # short bounce off upper range + VWAP align + simple reversal (close < open)
                if (not made_entry) and vwap_short_ok and (row["high"] >= dr_high - 0.00015) and (row["close"] < row["open"]):
                    sl = dr_high + 0.0005
                    tp = dr_low
                    units = pos_size(row["close"], sl)
                    if units > 0:
                        open_pos = Position(active_window, "short", -1, row["time"], row["close"], sl, tp, units)
                        trades_this_window += 1; made_entry = True

        # ========== MANAGEMENT: intrabar TP/SL ==========
        if open_pos:
            # Check SL/TP hits (conservative: SL before TP if both touched)
            hit_sl = (row["low"] <= open_pos.sl) if open_pos.dir == +1 else (row["high"] >= open_pos.sl)
            hit_tp = False
            if open_pos.tp is not None:
                hit_tp = (row["high"] >= open_pos.tp) if open_pos.dir == +1 else (row["low"] <= open_pos.tp)
            exit_price = None
            if hit_sl and hit_tp:
                exit_price = open_pos.sl
            elif hit_sl:
                exit_price = open_pos.sl
            elif hit_tp:
                exit_price = open_pos.tp
            if exit_price is not None:
                pnl = (exit_price - open_pos.entry) * open_pos.dir * open_pos.units
                balance += pnl; equity = balance
                trades.append({
                    **open_pos.__dict__,
                    "exit_time": row["time"], "exit": exit_price, "pnl": pnl
                })
                open_pos = None

        equity_curve.append({"time": row["time"], "equity": equity})

    # Exit any remaining position at final close
    if open_pos:
        exit_price = df.iloc[-1]["close"]
        pnl = (exit_price - open_pos.entry) * open_pos.dir * open_pos.units
        balance += pnl; equity = balance
        trades.append({**open_pos.__dict__, "exit_time": df.iloc[-1]["time"], "exit": exit_price, "pnl": pnl})
        open_pos = None

    summary = {
        "pair": PAIR,
        "start_balance": start_bal,
        "end_balance": balance,
        "net_pnl": balance - start_bal,
        "return_pct": 100.0 * (balance / start_bal - 1.0),
        "n_trades": len(trades),
    }
    return {"summary": summary, "trades": trades, "equity": equity_curve}

# ========== CLI / main ==========
def load_source(args) -> pd.DataFrame:
    if args.source == "alpha":
        return fetch_alpha(args.from_date, args.to_date)
    if args.source == "twelve":
        return fetch_twelve(args.from_date, args.to_date)
    if args.source == "oanda":
        return fetch_oanda(args.from_date, args.to_date)
    if args.source == "csv":
        if not args.csv:
            raise SystemExit("--csv path/to/file.csv is required when --source csv")
        df = pd.read_csv(args.csv)
        if "time" not in df.columns:
            raise SystemExit("CSV must have columns: time, open, high, low, close, volume")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        return df
    raise SystemExit("Unknown --source")

def main():
    ap = argparse.ArgumentParser(description="EURUSD M5 DR/IDR + ORB backtest")
    ap.add_argument("--from", dest="from_date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--to", dest="to_date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--source", choices=["alpha","twelve","oanda","csv"], default="csv")
    ap.add_argument("--csv", help="CSV path if --source csv")
    ap.add_argument("--events_csv", help="Optional CSV of news events with a 'time' (UTC) column", default=None)
    ap.add_argument("--no_embargo", action="store_true", help="Disable news embargo window")
    args = ap.parse_args()

    df = load_source(args)
    df = df[["time","open","high","low","close","volume"]].copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)

    events = load_events_csv(args.events_csv) if args.events_csv else []
    res = run_backtest(
        df,
        risk_pct=RISK_PCT,
        max_daily_loss_pct=MAX_DAILY_LOSS,
        dr_cap=DR_TRADES_CAP,
        orb_cap=ORB_TRADES_CAP,
        events=events,
        use_embargo=(not args.no_embargo)
    )

    # Save outputs
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(res["trades"]).to_csv("results/trades.csv", index=False)
    pd.DataFrame(res["equity"]).to_csv("results/equity_curve.csv", index=False)
    with open("results/backtest_summary.json", "w") as f:
        json.dump(res["summary"], f, indent=2)
    print(json.dumps(res["summary"], indent=2))

if __name__ == "__main__":
    main()
