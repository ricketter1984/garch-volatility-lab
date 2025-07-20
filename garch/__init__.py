"""
GARCH Volatility Lab Package

This package provides tools for volatility forecasting using GARCH models,
backtesting strategies with volatility overlays, and computing volatility metrics.

Modules:
    forecast: GARCH volatility modeling using the arch library
    backtest: Strategy backtesting with GARCH overlays
    metrics: Volatility statistics, drawdowns, and regime shift detection
"""

from .forecast import GARCHForecaster
from .backtest import GARCHBacktester
from .metrics import VolatilityMetrics

__version__ = "0.1.0"
__author__ = "GARCH Volatility Lab"

__all__ = ["GARCHForecaster", "GARCHBacktester", "VolatilityMetrics"] 