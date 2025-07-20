#!/usr/bin/env python3
"""
GARCH Model Diagnostics Module

This module provides comprehensive diagnostics for GARCH volatility models,
including model fitting, comparison, and visualization.

Key Features:
    - ARCH(1) and GARCH(1,1) model fitting
    - Model comparison and diagnostics
    - Comprehensive visualization of price, returns, residuals, and volatility
    - Volatility forecasting with confidence intervals
    - CLI interface for easy execution

Usage:
    python garch/diagnostics.py --symbol SPY --period 5y
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import argparse
import warnings
from typing import Tuple, Optional, Dict, Any
from arch import arch_model
import os

# Import CME scraper
try:
    import sys
    import os
    # Add parent directory to path to find cme_scraper
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from cme_scraper import get_price_series
    CME_AVAILABLE = True
except ImportError:
    print("⚠ CME scraper not available, falling back to yfinance")
    import yfinance as yf
    CME_AVAILABLE = False

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


def load_price_data(symbol: str, period: str = "5y") -> pd.DataFrame:
    """
    Load historical price data for a symbol using CME scraper or yfinance fallback.
    
    Args:
        symbol: Trading symbol
        period: Data period (e.g., "5y", "2y", "1y")
        
    Returns:
        DataFrame with price data
    """
    try:
        print(f"📊 Loading {period} of price data for {symbol}...")
        
        # Calculate date range from period
        end_date = datetime.now()
        if period.endswith('y'):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=years * 365)
        elif period.endswith('m'):
            months = int(period[:-1])
            start_date = end_date - timedelta(days=months * 30)
        else:
            days = int(period[:-1]) if period.endswith('d') else int(period)
            start_date = end_date - timedelta(days=days)
        
        if CME_AVAILABLE:
            # Try CME scraper first
            print(f"🔍 Attempting to fetch CME data for {symbol}...")
            prices = get_price_series(symbol, start_date, end_date)
            
            if not prices.empty:
                # Convert Series to DataFrame format for compatibility
                data = pd.DataFrame({'Close': prices})
                print(f"✓ Loaded {len(data)} days of CME data for {symbol}")
                return data
            else:
                print(f"⚠ CME data not available for {symbol}, falling back to yfinance")
        
        # Fallback to yfinance
        if not CME_AVAILABLE:
            print(f"📊 Using yfinance fallback for {symbol}...")
        else:
            print(f"📊 Using yfinance fallback for {symbol}...")
            
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Ensure we have sufficient data
        if len(data) < 100:
            raise ValueError(f"Insufficient data: {len(data)} observations (need >=100)")
        
        print(f"✓ Loaded {len(data)} days of data for {symbol}")
        return data
        
    except Exception as e:
        print(f"❌ Error loading data for {symbol}: {e}")
        return pd.DataFrame()


def calculate_returns(data: pd.DataFrame, method: str = "log") -> pd.Series:
    """
    Calculate returns from price data.
    
    Args:
        data: DataFrame with price data
        method: "log" for log returns or "pct" for percentage returns
        
    Returns:
        Series of returns
    """
    try:
        prices = data['Close'].dropna()
        
        if method == "log":
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        returns = returns.dropna()
        
        print(f"✓ Calculated {len(returns)} returns using {method} method")
        return returns
        
    except Exception as e:
        print(f"❌ Error calculating returns: {e}")
        return pd.Series()


def fit_garch_models(returns: pd.Series) -> Dict[str, Any]:
    """
    Fit ARCH(1) and GARCH(1,1) models to returns.
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary with fitted models and diagnostics
    """
    try:
        # Scale returns to percentage for better numerical stability
        scaled_returns = returns * 100
        
        print("🔧 Fitting ARCH(1) model...")
        # Fit ARCH(1) model
        arch_model_1 = arch_model(scaled_returns, vol='ARCH', p=1, mean='Constant')
        arch_fitted = arch_model_1.fit(disp='off')
        
        print("🔧 Fitting GARCH(1,1) model...")
        # Fit GARCH(1,1) model
        garch_model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, mean='Constant')
        garch_fitted = garch_model.fit(disp='off')
        
        # Calculate standardized residuals
        arch_resid = arch_fitted.resid / np.sqrt(arch_fitted.conditional_volatility)
        garch_resid = garch_fitted.resid / np.sqrt(garch_fitted.conditional_volatility)
        
        # Generate forecasts
        arch_forecast = arch_fitted.forecast(horizon=1)
        garch_forecast = garch_fitted.forecast(horizon=1)
        
        results = {
            'arch_model': arch_fitted,
            'garch_model': garch_fitted,
            'arch_resid': arch_resid,
            'garch_resid': garch_resid,
            'arch_forecast': arch_forecast,
            'garch_forecast': garch_forecast,
            'returns': returns
        }
        
        print("✓ Models fitted successfully")
        return results
        
    except Exception as e:
        print(f"❌ Error fitting models: {e}")
        return {}


def print_model_summaries(results: Dict[str, Any]) -> None:
    """
    Print detailed model summaries and diagnostics.
    
    Args:
        results: Dictionary with fitted models
    """
    try:
        print("\n" + "="*80)
        print("MODEL SUMMARIES")
        print("="*80)
        
        # ARCH(1) Summary
        print("\n📈 ARCH(1) MODEL SUMMARY")
        print("-" * 40)
        arch_model = results['arch_model']
        print(arch_model.summary())
        
        # GARCH(1,1) Summary
        print("\n📈 GARCH(1,1) MODEL SUMMARY")
        print("-" * 40)
        garch_model = results['garch_model']
        print(garch_model.summary())
        
        # Model Comparison
        print("\n📊 MODEL COMPARISON")
        print("-" * 40)
        print(f"ARCH(1) AIC:  {arch_model.aic:.4f}")
        print(f"GARCH(1,1) AIC: {garch_model.aic:.4f}")
        print(f"ARCH(1) BIC:  {arch_model.bic:.4f}")
        print(f"GARCH(1,1) BIC: {garch_model.bic:.4f}")
        
        # Volatility Forecasts
        print("\n🔮 VOLATILITY FORECASTS (Next Day)")
        print("-" * 40)
        arch_vol = np.sqrt(arch_model.forecast(horizon=1).variance.iloc[-1].values[0]) / 100
        garch_vol = np.sqrt(garch_model.forecast(horizon=1).variance.iloc[-1].values[0]) / 100
        print(f"ARCH(1) Forecast:  {arch_vol:.4f} ({arch_vol:.2%})")
        print(f"GARCH(1,1) Forecast: {garch_vol:.4f} ({garch_vol:.2%})")
        
        # Regime Classification
        print("\n🎯 VOLATILITY REGIME CLASSIFICATION")
        print("-" * 40)
        for model_name, vol in [("ARCH(1)", arch_vol), ("GARCH(1,1)", garch_vol)]:
            if vol > 0.025:
                regime = "HIGH"
            elif vol < 0.012:
                regime = "LOW"
            else:
                regime = "NORMAL"
            print(f"{model_name}: {regime} ({vol:.2%})")
            
    except Exception as e:
        print(f"❌ Error printing summaries: {e}")


def create_diagnostic_plots(results: Dict[str, Any], symbol: str, 
                           save_path: Optional[str] = None) -> None:
    """
    Create comprehensive diagnostic plots.
    
    Args:
        results: Dictionary with fitted models and data
        symbol: Trading symbol for plot titles
        save_path: Optional path to save plots
    """
    try:
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Price and Returns
        ax1 = plt.subplot(3, 2, 1)
        returns = results['returns']
        prices = returns.index.to_series().map(lambda x: (1 + returns).cumprod().iloc[-1] * 100)
        ax1.plot(returns.index, prices, 'b-', linewidth=1, alpha=0.7)
        ax1.set_title(f'{symbol} Price (Normalized)', fontweight='bold')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(returns.index, returns * 100, 'g-', linewidth=0.8, alpha=0.7)
        ax2.set_title(f'{symbol} Returns (%)', fontweight='bold')
        ax2.set_ylabel('Returns (%)')
        ax2.grid(True, alpha=0.3)
        
        # 2. Standardized Residuals
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(returns.index, results['arch_resid'], 'r-', linewidth=0.8, alpha=0.7, label='ARCH(1)')
        ax3.set_title('Standardized Residuals - ARCH(1)', fontweight='bold')
        ax3.set_ylabel('Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(3, 2, 4)
        ax4.plot(returns.index, results['garch_resid'], 'b-', linewidth=0.8, alpha=0.7, label='GARCH(1,1)')
        ax4.set_title('Standardized Residuals - GARCH(1,1)', fontweight='bold')
        ax4.set_ylabel('Residuals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 3. Conditional Volatility
        ax5 = plt.subplot(3, 2, 5)
        arch_vol = results['arch_model'].conditional_volatility / 100
        garch_vol = results['garch_model'].conditional_volatility / 100
        
        ax5.plot(returns.index, arch_vol, 'r-', linewidth=1, alpha=0.7, label='ARCH(1)')
        ax5.plot(returns.index, garch_vol, 'b-', linewidth=1, alpha=0.7, label='GARCH(1,1)')
        ax5.set_title('Conditional Volatility', fontweight='bold')
        ax5.set_ylabel('Volatility')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 4. Volatility Forecast Comparison
        ax6 = plt.subplot(3, 2, 6)
        
        # Generate multi-step forecasts for comparison
        arch_forecast = results['arch_model'].forecast(horizon=10)
        garch_forecast = results['garch_model'].forecast(horizon=10)
        
        arch_forecast_vol = np.sqrt(arch_forecast.variance.iloc[-1].values) / 100
        garch_forecast_vol = np.sqrt(garch_forecast.variance.iloc[-1].values) / 100
        
        forecast_dates = pd.date_range(returns.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')
        
        ax6.plot(forecast_dates, arch_forecast_vol, 'r-o', linewidth=2, markersize=4, label='ARCH(1)')
        ax6.plot(forecast_dates, garch_forecast_vol, 'b-s', linewidth=2, markersize=4, label='GARCH(1,1)')
        ax6.set_title('Volatility Forecast Comparison', fontweight='bold')
        ax6.set_ylabel('Forecasted Volatility')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Add overall title
        plt.suptitle(f'GARCH Diagnostics for {symbol}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Diagnostic plots saved to: {save_path}")
            except Exception as e:
                print(f"⚠ Could not save plots: {e}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")


def run_diagnostics(symbol: str = "SPY", period: str = "5y", 
                   save_plots: bool = True) -> Dict[str, Any]:
    """
    Run complete GARCH diagnostics for a symbol.
    
    Args:
        symbol: Trading symbol
        period: Data period
        save_plots: Whether to save diagnostic plots
        
    Returns:
        Dictionary with diagnostic results
    """
    print(f"🔍 Starting GARCH Diagnostics for {symbol}")
    print("=" * 60)
    
    # Load data
    data = load_price_data(symbol, period)
    if data.empty:
        return {}
    
    # Calculate returns
    returns = calculate_returns(data, method="log")
    if returns.empty:
        return {}
    
    # Fit models
    results = fit_garch_models(returns)
    if not results:
        return {}
    
    # Print summaries
    print_model_summaries(results)
    
    # Create plots
    save_path = f"logs/{symbol}_diagnostics.png" if save_plots else None
    create_diagnostic_plots(results, symbol, save_path)
    
    print("\n✅ GARCH diagnostics completed successfully!")
    return results


def main():
    """Main execution function with CLI interface."""
    parser = argparse.ArgumentParser(description="GARCH Model Diagnostics")
    parser.add_argument('--symbol', default='SPY', help='Trading symbol')
    parser.add_argument('--period', default='5y', help='Data period (e.g., 5y, 2y, 1y)')
    parser.add_argument('--no-save', action='store_true', help='Do not save plots')
    
    args = parser.parse_args()
    
    try:
        # Run diagnostics
        results = run_diagnostics(
            symbol=args.symbol,
            period=args.period,
            save_plots=not args.no_save
        )
        
        if not results:
            print("❌ Diagnostics failed. Check the error messages above.")
            return 1
            
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Diagnostics interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 