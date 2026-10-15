"""
Triple Moving Average (TMA) convergence / lateral consolidation setups.

Daily: 10 & 20 EMA compressed together and close to the 50 EMA, 150 SMA or
200 SMA. Weekly: 10 & 20 week SMA compressed near the 40 week SMA.

Variant 1 - lateral consolidation before the main move (no RS needed, ADR(20) > 2%).
Variant 2 - after a 30-100% move in 3 weeks - 3 months, price pulls back and
            moves sideways while the 10/20 catch up (needs RS, ADR(20) > 4%).

Liquidity/price filters: Volume(30) > 500k, price $0.50 - $150.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import indicators as ind


@dataclass
class SetupParams:
    min_price: float = 0.50
    max_price: float = 150.0
    min_vol30: float = 500_000
    v1_min_adr: float = 2.0
    v2_min_adr: float = 4.0
    min_consol_bars: int = 6        # daily and weekly lateral consolidation minimum
    tma_tol_pct: float = 3.0        # 10/20 within this % of the anchor MA (or 1x ADR, whichever is larger)
    v2_min_move_pct: float = 30.0
    v2_max_move_pct: float = 100.0  # a bigger prior move is still flagged, just noted
    v2_min_bars: int = 15           # ~3 weeks
    v2_max_bars: int = 65           # ~3 months
    options_above: float = 20.0


def consolidation_length(df: pd.DataFrame, tol_pct: float, max_bars: int = 80) -> int:
    """Longest trailing window whose closing range stays within ``tol_pct`` of its midpoint."""
    closes = df["Close"].values[-max_bars:]
    best = 0
    hi = lo = closes[-1]
    for n in range(1, len(closes) + 1):
        v = closes[-n]
        hi, lo = max(hi, v), min(lo, v)
        mid = (hi + lo) / 2.0
        if (hi - lo) / mid * 100.0 > tol_pct:
            break
        best = n
    return best


def tma_convergence(fast1: float, fast2: float, anchors: dict, tol_pct: float) -> Optional[str]:
    """Return the name of the closest anchor MA the two fast MAs are compressed against."""
    if any(np.isnan(x) for x in (fast1, fast2)):
        return None
    if abs(fast1 - fast2) / fast2 * 100.0 > tol_pct:
        return None
    best, best_d = None, None
    for name, a in anchors.items():
        if a is None or np.isnan(a):
            continue
        d = max(abs(fast1 - a), abs(fast2 - a)) / a * 100.0
        if d <= tol_pct and (best_d is None or d < best_d):
            best, best_d = name, d
    return best


def prior_move(df: pd.DataFrame, consol_bars: int, p: SetupParams) -> float:
    """Largest low->high run-up (%) in the 3wk-3mo window ending where the consolidation began."""
    end = len(df) - consol_bars
    if end <= p.v2_min_bars:
        return float("nan")
    window = df.iloc[max(0, end - p.v2_max_bars):end + 1]
    lows, highs = window["Low"].values, window["High"].values
    best = 0.0
    for i in range(len(window)):
        j0 = i + p.v2_min_bars // 3      # require some duration, not a one-day gap
        if j0 >= len(window):
            break
        peak = highs[j0:].max()
        best = max(best, (peak / lows[i] - 1.0) * 100.0)
    return best


def analyze_setup(daily: pd.DataFrame, weekly: pd.DataFrame, mansfield_rs: float,
                  p: SetupParams = SetupParams()) -> dict:
    close = daily["Close"]
    c = float(close.iloc[-1])
    adr = float(ind.adr_pct(daily, 20).iloc[-1])
    vol30 = float(daily["Volume"].rolling(30).mean().iloc[-1])

    e10, e20, e50 = (float(ind.ema(close, n).iloc[-1]) for n in (10, 20, 50))
    s150, s200 = (float(ind.sma(close, n).iloc[-1]) for n in (150, 200))
    tol_d = max(p.tma_tol_pct, adr if adr == adr else 0.0)
    tma_d = tma_convergence(e10, e20, {"50EMA": e50, "150SMA": s150, "200SMA": s200}, tol_d)

    wc = weekly["Close"]
    w10, w20, w40 = (float(ind.sma(wc, n).iloc[-1]) for n in (10, 20, 40))
    w_adr = float(ind.adr_pct(weekly, 10).iloc[-1]) if len(weekly) >= 10 else float("nan")
    tol_w = max(p.tma_tol_pct * 1.5, (w_adr / 2.0) if w_adr == w_adr else 0.0)
    tma_w = tma_convergence(w10, w20, {"40WSMA": w40}, tol_w)

    consol_d = consolidation_length(daily, max(10.0, 2.5 * adr) if adr == adr else 10.0)
    w_rng = float(((weekly["High"] / weekly["Low"] - 1) * 100).iloc[-10:].mean())
    consol_w = consolidation_length(weekly, max(15.0, 1.5 * w_rng), max_bars=52)

    sq_d = bool(ind.squeeze_on(daily).iloc[-5:].any())
    sq_w = bool(ind.squeeze_on(weekly).iloc[-3:].any())

    o = ind.obv(daily)
    obv_rising = bool(o.iloc[-1] > o.iloc[-21]) if len(o) > 21 else False
    mv = ind.macd_v(daily)
    pm = ind.pmo(close)

    liquid = p.min_price <= c <= p.max_price and vol30 > p.min_vol30
    lateral = consol_d >= p.min_consol_bars or consol_w >= p.min_consol_bars
    compressed = tma_d is not None or tma_w is not None
    run = prior_move(daily, max(consol_d, p.min_consol_bars), p)
    has_rs = mansfield_rs == mansfield_rs and mansfield_rs > 0

    v1 = liquid and lateral and compressed and adr > p.v1_min_adr
    v2 = (liquid and lateral and compressed and adr > p.v2_min_adr and has_rs
          and run == run and run >= p.v2_min_move_pct)

    return {
        "adr20_pct": adr, "vol30": vol30,
        "ema10": e10, "ema20": e20, "ema50": e50, "sma150": s150, "sma200": s200,
        "wsma10": w10, "wsma20": w20, "wsma40": w40,
        "tma_daily": tma_d or "", "tma_weekly": tma_w or "",
        "consol_days": consol_d, "consol_weeks": consol_w,
        "bk_squeeze_daily": sq_d, "bk_squeeze_weekly": sq_w,
        "obv_rising_20d": obv_rising,
        "macd_v": float(mv.iloc[-1]), "pmo": float(pm.iloc[-1]),
        "pmo_rising": bool(pm.iloc[-1] > pm.iloc[-2]),
        "prior_move_pct": run, "passes_filters": liquid,
        "variant1": v1, "variant2": v2,
        "instrument": "Options (check liquidity/spreads)" if c > p.options_above else "Shares",
    }
