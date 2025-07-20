# GARCH Volatility Lab

This repo contains research and modeling for volatility forecasting using GARCH theory and other statistical techniques. 
Focused on futures instruments like MES, MYM, MGC, and FX (M6E), it supports regime detection and strategy optimization 
in trading systems like ASSET Elite Turbo and MacroIntel.

## Overview

The GARCH Volatility Lab provides a comprehensive framework for:

- **Volatility Forecasting**: GARCH model implementation using the `arch` library
- **Strategy Backtesting**: Integration of volatility forecasts into trading strategies
- **Risk Management**: Volatility-based position sizing and risk metrics
- **Regime Detection**: Identification of volatility clusters and market regimes

## Project Structure

```
garch-volatility-lab/
├── garch/                     # Core GARCH implementation
│   ├── __init__.py           # Package initialization
│   ├── forecast.py           # GARCH volatility model using arch library
│   ├── backtest.py           # Strategy backtesting with GARCH overlays
│   └── metrics.py            # Volatility stats, drawdowns, regime shifts
├── notebooks/                # Jupyter notebooks for analysis
│   └── starter_garch_forecast.ipynb  # MES/MGC forecasting example
├── data/                     # Data storage
│   └── example_mgc.csv       # Sample data file
├── tests/                    # Unit tests
│   └── test_forecast.py      # Basic unit tests for forecast module
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd garch-volatility-lab
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Quick Start

### Basic GARCH Forecasting

```python
from garch.forecast import GARCHForecaster
import yfinance as yf

# Load data
ticker = yf.Ticker("SPY")
data = ticker.history(period="2y")
returns = data['Close'].pct_change().dropna()

# Fit GARCH model
forecaster = GARCHForecaster(vol_model='GARCH', p=1, q=1)
forecaster.fit(returns)

# Generate forecast
forecast = forecaster.forecast(horizon=1)
print(f"Next-day volatility: {forecast['volatility_forecast'][0]:.4f}")
print(f"Annualized volatility: {forecast['annualized_volatility'][0]:.4f}")
```

### Strategy Backtesting with GARCH

```python
from garch.backtest import GARCHBacktester
import pandas as pd

# Assume you have price data and trading signals
backtester = GARCHBacktester(initial_capital=100000)

# Backtest with GARCH-based position sizing
results = backtester.backtest_strategy(
    price_data=prices,
    signals=signals,
    garch_forecaster=forecaster,
    target_volatility=0.15,
    use_garch_sizing=True
)

print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

## Key Features

### 1. GARCH Forecasting (`garch.forecast`)
- Multiple GARCH model specifications (GARCH, EGARCH, GJR-GARCH)
- Volatility forecasting with confidence intervals
- Model diagnostics and parameter estimation
- Volatility clustering detection

### 2. Strategy Backtesting (`garch.backtest`)
- Volatility-targeted position sizing
- GARCH-based regime detection for strategy overlay
- Performance comparison with and without GARCH
- Transaction cost modeling

### 3. Risk Metrics (`garch.metrics`)
- Traditional metrics: Sharpe, Sortino, Calmar ratios
- Drawdown analysis and duration statistics
- Value at Risk (VaR) and Expected Shortfall
- Volatility regime classification

## Supported Instruments

The framework is designed for futures trading, particularly:

- **MES**: Micro E-mini S&P 500 futures
- **MYM**: Micro E-mini Dow Jones futures  
- **MGC**: Micro Gold futures
- **M6E**: Micro EUR/USD futures

*Note: Examples use ETF proxies (SPY, GLD) when direct futures data is not available.*

## Integration with Trading Systems

This lab is designed as a plugin/submodule for:

- **MacroIntel**: Macro trading intelligence platform
- **ASSET Elite Turbo**: Professional trading system
- Custom trading frameworks requiring volatility forecasting

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Or run specific tests:

```bash
python tests/test_forecast.py
```

## Examples

### Jupyter Notebooks

- `notebooks/starter_garch_forecast.ipynb`: Complete example loading MES/MGC data and generating volatility forecasts

### Sample Usage

1. **Volatility Forecasting**: Generate next-day volatility predictions
2. **Regime Detection**: Identify high/low volatility periods
3. **Position Sizing**: Scale positions based on forecasted volatility
4. **Strategy Enhancement**: Overlay GARCH signals on existing strategies

## Configuration

Key parameters can be configured via environment variables:

- `RISK_FREE_RATE`: Risk-free rate for Sharpe ratio calculation (default: 0.02)
- `TARGET_VOLATILITY`: Target portfolio volatility for position sizing (default: 0.15)
- `FORECAST_HORIZON`: Default forecast horizon in periods (default: 1)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

See LICENSE file for details.

## Future Development

- Real-time data integration
- Machine learning volatility models
- Multi-asset portfolio optimization
- Options pricing with GARCH volatility
- Integration with additional trading platforms
