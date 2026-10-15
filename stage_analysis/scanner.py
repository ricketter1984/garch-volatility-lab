"""
Orchestrates a full scan: stage per stock, setup flags, sector/theme tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from . import indicators as ind
from .data import to_weekly
from .sectors import basket_performance, etf_performance, returns_table
from .setups import SetupParams, analyze_setup
from .stages import StageParams, analyze
from .universe import Universe

# view name -> (description, row filter)
VIEWS = {
    "1to2": ("Stage 1 -> 2 transition candidates (late base / fresh breakout)",
             lambda d: d["substage"].isin(["1->2 Transition", "2A Breakout"])),
    "1": ("Stage 1 - basing", lambda d: d["stage"] == 1),
    "2": ("Stage 2 - advancing", lambda d: d["stage"] == 2),
    "2a": ("Stage 2A - early advance / breakout", lambda d: d["substage"].str.startswith("2A")),
    "3": ("Stage 3 - topping", lambda d: d["stage"] == 3),
    "4": ("Stage 4 - declining", lambda d: d["stage"] == 4),
    "v1": ("TMA Variant 1 setups", lambda d: d["variant1"].fillna(False).astype(bool)),
    "v2": ("TMA Variant 2 setups", lambda d: d["variant2"].fillna(False).astype(bool)),
    "all": ("All stocks", lambda d: pd.Series(True, index=d.index)),
}


@dataclass
class ScanResult:
    stocks: pd.DataFrame
    sector_etfs: pd.DataFrame
    theme_etfs: pd.DataFrame
    sector_baskets: pd.DataFrame
    theme_baskets: pd.DataFrame
    returns: pd.DataFrame
    as_of: pd.Timestamp

    def view(self, name: str) -> pd.DataFrame:
        _, fn = VIEWS[name]
        df = self.stocks[fn(self.stocks)]
        key = "transition_score" if name in ("1to2", "1") else "tma_rank"
        return df.sort_values([key, "RS_3M"], ascending=False, na_position="last")


def scan(prices: Dict[str, pd.DataFrame], universe: Universe,
         stage_params: StageParams = StageParams(),
         setup_params: SetupParams = SetupParams(),
         require_filters: bool = False) -> ScanResult:
    bench = prices.get(universe.benchmark)
    bench_w = to_weekly(bench)["Close"] if bench is not None else None

    rets = returns_table(prices)
    records = []
    for t in universe.stocks:
        daily = prices.get(t)
        if daily is None or len(daily) < 220:
            continue
        weekly = to_weekly(daily)
        rec = analyze(weekly, bench_w, daily, stage_params)
        if rec is None:
            continue
        if "tr_mansfield_rs" not in rec and bench_w is not None:
            rs = ind.mansfield_rs(weekly["Close"], bench_w, 52)
            rec["tr_mansfield_rs"] = float(rs.iloc[-1])
        rec.update(analyze_setup(daily, weekly, rec.get("tr_mansfield_rs", float("nan")), setup_params))
        rec["ticker"] = t
        rec["sector"] = universe.sectors[t]
        rec["themes"] = ", ".join(universe.themes.get(t, []))
        records.append(rec)

    stocks = pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()
    if not stocks.empty:
        stocks = stocks.join(rets, how="left")
        if universe.benchmark in rets.index:
            for p in ("1M", "3M", "6M"):
                stocks[f"RS_{p}"] = stocks[p] - rets.loc[universe.benchmark, p]
        stocks.rename(columns={"tr_mansfield_rs": "mansfield_rs"}, inplace=True)
        # simple ordering key for setup views: compression + squeeze + consolidation
        stocks["tma_rank"] = ((stocks["tma_daily"] != "").astype(int) * 2
                              + (stocks["tma_weekly"] != "").astype(int) * 2
                              + stocks["bk_squeeze_daily"].astype(int)
                              + stocks["bk_squeeze_weekly"].astype(int)
                              + stocks["consol_weeks"].clip(upper=12) / 6)
        if require_filters:
            stocks = stocks[stocks["passes_filters"]]

    stage_cols = stocks[["stage", "substage", "pct_from_ma30w"]] if not stocks.empty else pd.DataFrame()
    last = max(df.index[-1] for df in prices.values()) if prices else pd.Timestamp.now()

    return ScanResult(
        stocks=stocks,
        sector_etfs=etf_performance(rets, universe.sector_etfs, universe.benchmark),
        theme_etfs=etf_performance(rets, universe.theme_etfs, universe.benchmark),
        sector_baskets=basket_performance(rets, universe.sector_members(), stage_cols, universe.benchmark),
        theme_baskets=basket_performance(rets, universe.theme_members(), stage_cols, universe.benchmark),
        returns=rets,
        as_of=last,
    )
