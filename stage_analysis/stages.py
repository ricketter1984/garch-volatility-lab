"""
Weinstein stage classification on weekly bars.

Stage 1  Basing      - 30-week MA flattens after a decline, price chops around it.
Stage 2  Advancing   - price above a rising 30-week MA.
Stage 3  Topping     - MA flattens after an advance, price chops around it.
Stage 4  Declining   - price below a falling 30-week MA.

Flat-MA periods are resolved with a state machine: a flat MA after Stage 4
(or within Stage 1) is Stage 1; after Stage 2 (or within Stage 3) it is
Stage 3. A Stage 1 stock is additionally scored 0-100 on how ready it is to
transition into Stage 2 (see ``transition_score``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import indicators as ind

STAGE_NAMES = {1: "Stage 1 - Basing", 2: "Stage 2 - Advancing",
               3: "Stage 3 - Topping", 4: "Stage 4 - Declining"}


@dataclass
class StageParams:
    ma_len: int = 30            # Weinstein 30-week MA
    slope_lookback: int = 4     # weeks used to measure MA slope
    flat_pct: float = 1.5       # |MA change over lookback| below this = flat
    transition_min_score: float = 60.0
    extended_pct: float = 25.0  # Stage 2 price > this % above MA30 = extended
    early_weeks: int = 8        # first N weeks of Stage 2 = 2A


def classify_series(weekly: pd.DataFrame, p: StageParams = StageParams()) -> pd.DataFrame:
    """Return a frame with ma30, slope and integer stage for every weekly bar."""
    close = weekly["Close"]
    ma = ind.sma(close, p.ma_len)
    slope = ind.pct_slope(ma, p.slope_lookback)

    stages = np.full(len(weekly), np.nan)
    prev: Optional[int] = None
    for i in range(len(weekly)):
        m, s, c = ma.iloc[i], slope.iloc[i], close.iloc[i]
        if np.isnan(m) or np.isnan(s):
            continue
        if prev is None:
            # seed from where price sits relative to the MA a year ago
            ref = ma.iloc[max(0, i - 26)]
            prev = 2 if (not np.isnan(ref) and m > ref) else 4

        above = c > m
        rising, falling = s > p.flat_pct, s < -p.flat_pct

        if above and rising:
            st = 2
        elif (not above) and falling:
            st = 4
        elif prev == 2 and rising:
            st = 2                      # pullback under a still-rising MA
        elif prev == 4 and falling:
            st = 4 if not above else 1  # first push above a falling MA = early base
        elif prev in (4, 1):
            st = 1
        else:                           # prev in (2, 3)
            st = 3
        stages[i] = st
        prev = st

    return pd.DataFrame({"close": close, "ma30": ma, "ma_slope": slope, "stage": stages},
                        index=weekly.index)


def _run_length(stages: pd.Series) -> int:
    s = stages.dropna()
    if s.empty:
        return 0
    last = s.iloc[-1]
    n = 0
    for v in reversed(s.values):
        if v != last:
            break
        n += 1
    return n


def transition_score(weekly: pd.DataFrame, cls: pd.DataFrame, bench_weekly: Optional[pd.Series],
                     daily: Optional[pd.DataFrame], base_weeks: int,
                     p: StageParams = StageParams()) -> Dict[str, float]:
    """Score (0-100) how close a basing stock is to a Stage 1 -> 2 transition."""
    close, vol = weekly["Close"], weekly["Volume"]
    c = float(close.iloc[-1])
    ma30 = float(cls["ma30"].iloc[-1])
    slope = float(cls["ma_slope"].iloc[-1])
    w10, w20, w40 = (ind.sma(close, n).iloc[-1] for n in (10, 20, 40))

    lookback = int(min(max(base_weeks, 6), 52))
    base = weekly.iloc[-lookback - 1:-1] if len(weekly) > lookback + 1 else weekly.iloc[:-1]
    pivot = float(base["High"].max()) if not base.empty else float("nan")
    dist_pivot = (pivot / c - 1.0) * 100.0 if pivot == pivot else float("nan")

    pts: Dict[str, float] = {}
    pts["base_length"] = 10.0 * min(base_weeks, 26) / 26.0 if base_weeks >= 6 else 0.0
    pts["ma_flattening"] = 10.0 if slope >= 0 else (5.0 if slope > -p.flat_pct else 0.0)
    pts["price_vs_ma"] = (10.0 if c > ma30 else 0.0) + (5.0 if c > w10 else 0.0)
    pts["near_pivot"] = 15.0 if dist_pivot <= 5 else (8.0 if dist_pivot <= 10 else 0.0)

    tma_spread = float("nan")
    if not np.isnan(w40):
        tma_spread = (max(w10, w20, w40) - min(w10, w20, w40)) / w40 * 100.0
        pts["tma_convergence"] = 15.0 if tma_spread <= 5 else (8.0 if tma_spread <= 8 else 0.0)
    else:
        pts["tma_convergence"] = 0.0

    sq_w = bool(ind.squeeze_on(weekly).iloc[-3:].any())
    sq_d = bool(ind.squeeze_on(daily).iloc[-5:].any()) if daily is not None else False
    pts["squeeze"] = 10.0 if (sq_w or sq_d) else 0.0

    o = ind.obv(weekly)
    obv_up = o.iloc[-1] > o.iloc[-11] if len(o) > 11 else False
    obv_high = o.iloc[-1] >= o.iloc[-26:].max() * 0.98 if len(o) >= 26 else False
    pts["obv"] = (5.0 if obv_up else 0.0) + (5.0 if obv_high else 0.0)

    rs_now = rs_chg = float("nan")
    if bench_weekly is not None:
        rs = ind.mansfield_rs(close, bench_weekly, 52)
        if rs.notna().sum() > 5:
            rs_now, rs_chg = float(rs.iloc[-1]), float(rs.iloc[-1] - rs.iloc[-5])
            pts["rel_strength"] = (5.0 if rs_chg > 0 else 0.0) + (5.0 if rs_now > -5 else 0.0)
    pts.setdefault("rel_strength", 0.0)

    base_vol = float(base["Volume"].mean()) if not base.empty else float("nan")
    vol_ratio = float(vol.iloc[-4:].mean() / base_vol) if base_vol else float("nan")
    pts["volume_expansion"] = 5.0 if vol_ratio > 1.2 else 0.0

    return {"score": round(sum(pts.values()), 1), "pivot": pivot, "dist_to_pivot_pct": dist_pivot,
            "tma_spread_pct": tma_spread, "squeeze_weekly": sq_w, "squeeze_daily": sq_d,
            "mansfield_rs": rs_now, "rs_4w_change": rs_chg, "vol_ratio_4w": vol_ratio,
            **{f"pts_{k}": v for k, v in pts.items()}}


def analyze(weekly: pd.DataFrame, bench_weekly: Optional[pd.Series] = None,
            daily: Optional[pd.DataFrame] = None, p: StageParams = StageParams()) -> Optional[dict]:
    """Classify the latest weekly bar and return a summary record (None if too little data)."""
    cls = classify_series(weekly, p)
    st_series = cls["stage"]
    if st_series.dropna().empty:
        return None

    stage = int(st_series.dropna().iloc[-1])
    weeks_in = _run_length(st_series)
    prior = st_series.dropna().iloc[:-weeks_in]
    prev_stage = int(prior.iloc[-1]) if not prior.empty else None
    c = float(weekly["Close"].iloc[-1])
    ma30 = float(cls["ma30"].iloc[-1])
    ext = (c / ma30 - 1.0) * 100.0

    rec = {"stage": stage, "stage_name": STAGE_NAMES[stage], "weeks_in_stage": weeks_in,
           "prev_stage": prev_stage, "close": c, "ma30w": ma30, "ma30w_slope_pct": float(cls["ma_slope"].iloc[-1]),
           "pct_from_ma30w": ext}

    # Stage 1 base length counts back through the preceding Stage 1 run when a
    # stock has just broken into Stage 2, so fresh breakouts are scored too.
    base_weeks = weeks_in if stage == 1 else 0
    if stage == 2 and prev_stage == 1:
        base_weeks = _run_length(prior)

    if stage == 1 or (stage == 2 and prev_stage == 1 and weeks_in <= 4):
        tr = transition_score(weekly, cls, bench_weekly, daily, base_weeks, p)
        rec.update({f"tr_{k}" if k != "score" else "transition_score": v for k, v in tr.items()})
    else:
        rec["transition_score"] = float("nan")

    if stage == 1:
        sub = "1->2 Transition" if rec["transition_score"] >= p.transition_min_score else "1 Basing"
    elif stage == 2:
        if prev_stage == 1 and weeks_in <= 4:
            sub = "2A Breakout"
        elif weeks_in <= p.early_weeks:
            sub = "2A Early"
        elif ext > p.extended_pct:
            sub = "2B Extended"
        else:
            sub = "2 Advancing"
    elif stage == 3:
        sub = "3 Topping"
    else:
        sub = "4A Breakdown" if weeks_in <= 4 else "4 Declining"
    rec["substage"] = sub
    return rec
