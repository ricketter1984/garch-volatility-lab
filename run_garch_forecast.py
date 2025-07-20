#!/usr/bin/env python3
"""
GARCH Volatility Forecast CLI Tool

This script provides a command-line interface for generating GARCH volatility forecasts.
It's designed for integration with MacroIntel and other client tools.

Usage:
    python run_garch_forecast.py --symbol SPY --period 1y
    python run_garch_forecast.py --symbol MES=F --period 2y
    python run_garch_forecast.py  # Uses defaults: SPY, 1y

Author: GARCH Volatility Lab
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple

# Add current directory to path for garch module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from garch.forecast import forecast_next_day_volatility
except ImportError as e:
    print(f"ERROR: Could not import GARCH forecast module: {e}")
    print("Make sure you're running from the garch-volatility-lab directory")
    print("and that all dependencies are installed: pip install -r requirements.txt")
    sys.exit(1)


def interpret_volatility(vol: float) -> str:
    """
    Interpret volatility level based on thresholds.
    
    Args:
        vol: Daily volatility as decimal (e.g., 0.025 for 2.5%)
        
    Returns:
        Regime classification: 'HIGH', 'LOW', or 'NORMAL'
    """
    if vol > 0.025:
        return "HIGH"
    elif vol < 0.012:
        return "LOW"
    else:
        return "NORMAL"


def setup_logging_directory() -> Path:
    """
    Create logs directory if it doesn't exist.
    
    Returns:
        Path to logs directory
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def log_forecast_to_csv(
    symbol: str, 
    period: str, 
    forecast_vol: float, 
    regime: str,
    csv_path: Path
) -> None:
    """
    Append forecast results to CSV log file.
    
    Args:
        symbol: Trading symbol
        period: Data period used
        forecast_vol: Forecasted volatility
        regime: Volatility regime classification
        csv_path: Path to CSV log file
    """
    # Check if file exists to determine if we need headers
    file_exists = csv_path.exists()
    
    # Prepare row data
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row_data = [timestamp, symbol, period, f"{forecast_vol:.6f}", regime]
    
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write headers if file is new
            if not file_exists:
                headers = ['date', 'symbol', 'period', 'forecast_vol', 'regime']
                writer.writerow(headers)
            
            # Write data row
            writer.writerow(row_data)
            
        print(f"✓ Results logged to: {csv_path}")
        
    except Exception as e:
        print(f"⚠ Warning: Could not log to CSV file: {e}")


def format_output(symbol: str, period: str, forecast_vol: float, regime: str) -> None:
    """
    Print formatted forecast results to console.
    
    Args:
        symbol: Trading symbol
        period: Data period used
        forecast_vol: Forecasted volatility
        regime: Volatility regime classification
    """
    annualized_vol = forecast_vol * (252 ** 0.5)
    
    print("=" * 60)
    print("GARCH VOLATILITY FORECAST")
    print("=" * 60)
    print(f"Symbol:               {symbol}")
    print(f"Data Period:          {period}")
    print(f"Timestamp:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)
    print(f"Forecasted Vol:       {forecast_vol:.4f} ({forecast_vol:.2%})")
    print(f"Annualized Vol:       {annualized_vol:.4f} ({annualized_vol:.1%})")
    print(f"Regime:               {regime}")
    print("-" * 60)
    
    # Add interpretation guidance
    if regime == "HIGH":
        print("📈 HIGH VOLATILITY: Consider reducing position sizes")
        print("   Threshold: >2.5% daily volatility")
    elif regime == "LOW":
        print("📉 LOW VOLATILITY: Consider increasing position sizes")
        print("   Threshold: <1.2% daily volatility")
    else:
        print("📊 NORMAL VOLATILITY: Standard position sizing")
        print("   Range: 1.2% - 2.5% daily volatility")
    
    print("=" * 60)


def parse_arguments() -> Tuple[str, str, bool]:
    """
    Parse command line arguments.
    
    Returns:
        Tuple of (symbol, period, verbose)
    """
    parser = argparse.ArgumentParser(
        description="Generate GARCH volatility forecasts for trading symbols",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_garch_forecast.py                           # SPY with 1y data
  python run_garch_forecast.py --symbol GLD              # Gold ETF with 1y data  
  python run_garch_forecast.py --symbol MES=F --period 2y # Micro E-mini with 2y data
  python run_garch_forecast.py --symbol EURUSD=X --period 6mo # FX with 6mo data

Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        """
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='SPY',
        help='Trading symbol to forecast (default: SPY)'
    )
    
    parser.add_argument(
        '--period', '-p',
        type=str,
        default='1y',
        help='Historical data period (default: 1y)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output for debugging'
    )
    
    args = parser.parse_args()
    return args.symbol, args.period, args.verbose


def main():
    """Main execution function."""
    try:
        # Parse command line arguments
        symbol, period, verbose = parse_arguments()
        
        if verbose:
            print(f"Starting GARCH forecast for {symbol} using {period} of data...")
        
        # Setup logging directory
        logs_dir = setup_logging_directory()
        csv_path = logs_dir / "garch_forecast_log.csv"
        
        # Generate GARCH forecast
        if verbose:
            print("Fitting GARCH model and generating forecast...")
            
        forecast_vol = forecast_next_day_volatility(symbol=symbol, period=period)
        
        # Check if forecast was successful
        if forecast_vol == 0.0:
            print(f"❌ ERROR: Could not generate forecast for {symbol}")
            print("Possible issues:")
            print("  - Symbol not found or invalid")
            print("  - Insufficient historical data")
            print("  - GARCH model failed to converge")
            print("  - Network/data provider issues")
            sys.exit(1)
        
        # Interpret volatility regime
        regime = interpret_volatility(forecast_vol)
        
        # Display results
        format_output(symbol, period, forecast_vol, regime)
        
        # Log results to CSV
        log_forecast_to_csv(symbol, period, forecast_vol, regime, csv_path)
        
        if verbose:
            print(f"Forecast completed successfully for {symbol}")
            
    except KeyboardInterrupt:
        print("\n❌ Forecast interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ ERROR: Unexpected error occurred: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 