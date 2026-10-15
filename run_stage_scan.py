#!/usr/bin/env python3
"""
Stage Analysis Scanner - CLI

Examples:
    python run_stage_scan.py                         # Stage 1->2 candidates + sector/theme tables
    python run_stage_scan.py --stage 2a              # early Stage 2 advances
    python run_stage_scan.py --stage 4 --theme semis # declining AI semis
    python run_stage_scan.py --stage v2 --filters    # Variant 2 setups that pass price/volume filters
    python run_stage_scan.py --universe my_list.csv --html reports/scan.html

Stage views: 1to2, 1, 2, 2a, 3, 4, v1, v2, all
"""

import argparse
import os
import sys

import pandas as pd

from stage_analysis import VIEWS, SetupParams, StageParams, default_universe, load_universe_csv, scan
from stage_analysis.data import load_prices
from stage_analysis.report import write_html

STOCK_PRINT = ["substage", "transition_score", "weeks_in_stage", "close", "pct_from_ma30w",
               "tr_dist_to_pivot_pct", "mansfield_rs", "RS_3M", "adr20_pct", "tma_daily",
               "tma_weekly", "consol_weeks", "bk_squeeze_daily", "variant1", "variant2", "themes"]
GROUP_PRINT = ["1W", "1M", "3M", "6M", "YTD", "1Y", "RS_1M", "RS_3M", "Rotation", "%S1", "%S2",
               "%S4", "1->2 Cands"]


def show(title, df, cols, n):
    print(f"\n=== {title} ===")
    if df is None or df.empty:
        print("(none)")
        return
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(n).to_string(float_format=lambda v: f"{v:,.1f}"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="1to2", choices=list(VIEWS), help="stage view to list")
    ap.add_argument("--theme", nargs="*", help="limit stocks to themes matching these words")
    ap.add_argument("--universe", help="CSV with ticker,sector,themes (merged with built-in list)")
    ap.add_argument("--only-universe", action="store_true", help="use only the CSV universe")
    ap.add_argument("--filters", action="store_true", help="require price $0.50-150 and Vol(30) > 500k")
    ap.add_argument("--period", default="3y", help="history to download (default 3y)")
    ap.add_argument("--refresh", action="store_true", help="ignore the price cache")
    ap.add_argument("--top", type=int, default=40, help="rows to print per table")
    ap.add_argument("--min-score", type=float, default=60.0, help="1->2 transition score threshold")
    ap.add_argument("--html", default="reports/stage_scan.html", help="HTML report path ('' to skip)")
    ap.add_argument("--csv", help="also write the full stock table to this CSV")
    args = ap.parse_args(argv)

    if args.universe and args.only_universe:
        uni = load_universe_csv(args.universe)
    else:
        uni = default_universe()
        if args.universe:
            uni = load_universe_csv(args.universe, uni)
    uni = uni.filter_themes(args.theme)
    if not uni.stocks:
        sys.exit("No stocks match that theme filter.")

    print(f"Loading {len(uni.all_tickers)} tickers ({args.period}) ...")
    prices = load_prices(uni.all_tickers, period=args.period, refresh=args.refresh)
    if uni.benchmark not in prices:
        sys.exit(f"Could not load benchmark {uni.benchmark}; check your connection.")

    res = scan(prices, uni, StageParams(transition_min_score=args.min_score), SetupParams(),
               require_filters=args.filters)
    print(f"Data as of {res.as_of:%Y-%m-%d} | scanned {len(res.stocks)} stocks")
    if not res.stocks.empty:
        print("Stage counts:", res.stocks["substage"].value_counts().to_dict())

    show(VIEWS[args.stage][0], res.view(args.stage), STOCK_PRINT, args.top)
    show("Themes (median member return %, RS vs SPY, stage breadth)", res.theme_baskets, GROUP_PRINT, args.top)
    show("Sectors (stock baskets)", res.sector_baskets, GROUP_PRINT, args.top)
    show("Sector ETFs", res.sector_etfs, ["Group"] + GROUP_PRINT, args.top)
    show("Theme ETFs", res.theme_etfs, ["Group"] + GROUP_PRINT, args.top)

    if args.csv:
        res.stocks.to_csv(args.csv)
        print(f"\nWrote {args.csv}")
    if args.html:
        os.makedirs(os.path.dirname(args.html) or ".", exist_ok=True)
        write_html(res, args.html, default_view=args.stage)
        print(f"Wrote {args.html} - open it in a browser to switch stages/themes interactively.")


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    main()
