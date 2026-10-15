"""
Price data loading with a simple on-disk cache.

Daily OHLCV comes from yfinance (split/dividend adjusted). Weekly bars are
resampled locally (week ending Friday) so daily and weekly stay consistent.
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Dict, Iterable, List, Optional

import pandas as pd

OHLCV = ["Open", "High", "Low", "Close", "Volume"]
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "stage_cache")


def to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample daily OHLCV to weekly bars ending Friday."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    w = daily[OHLCV].resample("W-FRI").agg(agg)
    return w.dropna(subset=["Close"])


def _clean(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = df[[c for c in OHLCV if c in df.columns]].dropna(subset=["Close"])
    if len(df) < 60:
        return None
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def download_daily(tickers: Iterable[str], period: str = "3y",
                   chunk_size: int = 80, pause: float = 1.0) -> Dict[str, pd.DataFrame]:
    """Download daily bars for ``tickers`` via yfinance in chunks."""
    import yfinance as yf

    tickers = list(dict.fromkeys(t.upper() for t in tickers))
    out: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        raw = yf.download(chunk, period=period, interval="1d", auto_adjust=True,
                          group_by="ticker", progress=False, threads=True)
        for t in chunk:
            try:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            df = _clean(df)
            if df is not None:
                out[t] = df
        if i + chunk_size < len(tickers):
            time.sleep(pause)
    return out


def load_prices(tickers: List[str], period: str = "3y", cache_dir: str = DEFAULT_CACHE_DIR,
                max_age_hours: float = 12.0, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Load daily prices, using a pickle cache younger than ``max_age_hours``."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"daily_{period}.pkl")
    cached: Dict[str, pd.DataFrame] = {}
    if os.path.exists(path) and not refresh:
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        if age_h <= max_age_hours:
            with open(path, "rb") as fh:
                cached = pickle.load(fh)

    missing = [t for t in tickers if t not in cached]
    if missing:
        cached.update(download_daily(missing, period=period))
        with open(path, "wb") as fh:
            pickle.dump(cached, fh)
    return {t: cached[t] for t in tickers if t in cached}
