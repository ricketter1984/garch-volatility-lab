"""
Sector and theme performance over multiple lookbacks, plus stage breadth.

Groups are measured two ways:
  * ETF proxies (sector SPDRs, thematic ETFs)
  * Baskets built from the scanned stocks (median member return), which also
    gives stage breadth: what % of the group sits in each stage.

Rotation quadrant (RRG-style, vs the benchmark):
  Leading    1M RS > 0 and 3M RS > 0
  Improving  1M RS > 0 and 3M RS <= 0   <- money rotating in
  Weakening  1M RS <= 0 and 3M RS > 0   <- money rotating out
  Lagging    1M RS <= 0 and 3M RS <= 0
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import indicators as ind

PERIODS = {"1W": 5, "1M": 21, "3M": 63, "6M": 126, "1Y": 252}
PERIOD_ORDER = ["1W", "1M", "3M", "6M", "YTD", "1Y"]


def returns_table(prices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for t, df in prices.items():
        c = df["Close"]
        r = {k: ind.period_return(c, n) for k, n in PERIODS.items()}
        r["YTD"] = ind.ytd_return(c)
        rows[t] = r
    return pd.DataFrame.from_dict(rows, orient="index")[PERIOD_ORDER]


def _quadrant(rs1m: float, rs3m: float) -> str:
    if np.isnan(rs1m) or np.isnan(rs3m):
        return ""
    if rs1m > 0:
        return "Leading" if rs3m > 0 else "Improving"
    return "Weakening" if rs3m > 0 else "Lagging"


def _finish(df: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    for p in PERIOD_ORDER:
        df[f"RS_{p}"] = df[p] - bench[p]
    df["Rotation"] = [_quadrant(a, b) for a, b in zip(df["RS_1M"], df["RS_3M"])]
    # composite rank: weight recent periods more
    w = {"1M": 0.4, "3M": 0.35, "6M": 0.25}
    df["Score"] = sum(df[p].rank(pct=True) * wt for p, wt in w.items()) * 100
    return df.sort_values("Score", ascending=False)


def etf_performance(rets: pd.DataFrame, etfs: Dict[str, str], benchmark: str) -> pd.DataFrame:
    have = [t for t in etfs if t in rets.index]
    if not have or benchmark not in rets.index:
        return pd.DataFrame()
    df = rets.loc[have].copy()
    df.insert(0, "Group", [etfs[t] for t in have])
    df.index.name = "ETF"
    return _finish(df, rets.loc[benchmark])


def basket_performance(rets: pd.DataFrame, members: Dict[str, List[str]], stages: pd.DataFrame,
                       benchmark: str) -> pd.DataFrame:
    """Median-member returns plus stage breadth for each group of stocks."""
    if benchmark not in rets.index:
        return pd.DataFrame()
    rows = {}
    for name, tickers in members.items():
        have = [t for t in tickers if t in rets.index]
        if not have:
            continue
        r = rets.loc[have].median().to_dict()
        r["N"] = len(have)
        st = stages.reindex(have)
        for s in (1, 2, 3, 4):
            r[f"%S{s}"] = float((st["stage"] == s).mean() * 100) if "stage" in st else np.nan
        if "substage" in st:
            r["1->2 Cands"] = int((st["substage"] == "1->2 Transition").sum())
            r["%Above30W"] = float((st["pct_from_ma30w"] > 0).mean() * 100)
        rows[name] = r
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "Group"
    return _finish(df, rets.loc[benchmark])
