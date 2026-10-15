"""
Stage Analysis Scanner

Weinstein stage classification (weekly, 30-week MA) with a Stage 1 -> 2
transition score, TMA-convergence / lateral-consolidation setup flags, and
sector/theme performance and rotation over multiple lookbacks.

Modules:
    universe: sector/theme ETF lists and thematic stock baskets
    data: yfinance download, caching, weekly resampling
    indicators: MAs, ADR, OBV, BB/KC squeeze, Mansfield RS, MACD-V, PMO
    stages: stage classifier and transition score
    setups: TMA convergence, Variant 1 / Variant 2 detection, filters
    sectors: multi-period sector/theme returns, RS, rotation, stage breadth
    scanner: end-to-end scan and stage views
    report: interactive HTML report
"""

from .scanner import VIEWS, ScanResult, scan
from .setups import SetupParams
from .stages import StageParams
from .universe import Universe, default_universe, load_universe_csv

__all__ = ["scan", "ScanResult", "VIEWS", "StageParams", "SetupParams",
           "Universe", "default_universe", "load_universe_csv"]
