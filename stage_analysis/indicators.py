"""
Indicator helpers used by the stage classifier and setup detector.

All functions take pandas Series/DataFrames indexed by date and return
Series aligned to the input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def pct_slope(s: pd.Series, lookback: int) -> pd.Series:
    """Percent change of a series over ``lookback`` bars (used for MA slope)."""
    return (s / s.shift(lookback) - 1.0) * 100.0


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["Close"].shift(1)
    return pd.concat([df["High"] - df["Low"], (df["High"] - prev).abs(),
                      (df["Low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 20) -> pd.Series:
    return true_range(df).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def adr_pct(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Average Daily Range % (High/Low - 1) over ``n`` bars."""
    return ((df["High"] / df["Low"] - 1.0) * 100.0).rolling(n, min_periods=n).mean()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Close"].diff()).fillna(0.0)
    return (direction * df["Volume"]).cumsum()


def squeeze_on(df: pd.DataFrame, n: int = 20, bb_mult: float = 2.0,
               kc_mult: float = 1.5) -> pd.Series:
    """Bollinger Bands inside Keltner Channels (TTM-style squeeze)."""
    mid = sma(df["Close"], n)
    sd = df["Close"].rolling(n, min_periods=n).std(ddof=0)
    rng = atr(df, n)
    bb_up, bb_lo = mid + bb_mult * sd, mid - bb_mult * sd
    kc_up, kc_lo = mid + kc_mult * rng, mid - kc_mult * rng
    return (bb_up < kc_up) & (bb_lo > kc_lo)


def mansfield_rs(close: pd.Series, bench: pd.Series, n: int = 52) -> pd.Series:
    """Mansfield relative strength: (RS / SMA(RS, n) - 1) * 100. Above 0 = outperforming."""
    bench = bench.reindex(close.index).ffill()
    rs = close / bench
    return (rs / sma(rs, n) - 1.0) * 100.0


def macd_v(df: pd.DataFrame, fast: int = 12, slow: int = 26, atr_n: int = 26) -> pd.Series:
    """MACD-V (Spiroglou): MACD normalised by ATR, x100."""
    m = ema(df["Close"], fast) - ema(df["Close"], slow)
    return m / atr(df, atr_n) * 100.0


def pmo(close: pd.Series, n1: int = 35, n2: int = 20) -> pd.Series:
    """DecisionPoint Price Momentum Oscillator."""
    roc = close.pct_change() * 100.0
    s1 = roc.ewm(alpha=2.0 / n1, adjust=False).mean() * 10.0
    return s1.ewm(alpha=2.0 / n2, adjust=False).mean()


def period_return(close: pd.Series, bars: int) -> float:
    if len(close) <= bars:
        return float("nan")
    return float((close.iloc[-1] / close.iloc[-1 - bars] - 1.0) * 100.0)


def ytd_return(close: pd.Series) -> float:
    last = close.index[-1]
    prior = close[close.index < pd.Timestamp(year=last.year, month=1, day=1)]
    if prior.empty:
        return float("nan")
    return float((close.iloc[-1] / prior.iloc[-1] - 1.0) * 100.0)
