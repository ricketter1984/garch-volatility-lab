#!/usr/bin/env python3
"""
Generate Historical GARCH Forecasts from Trade Journal

This script loads trade journal data and generates GARCH volatility forecasts
for each unique (symbol, date) pair to ensure complete coverage for overlay analysis.

Usage:
    python generate_forecasts_from_trades.py

Author: GARCH Volatility Lab
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os
from pathlib import Path
from typing import List, Tuple, Optional
from arch import arch_model

# Add current directory to path for garch module imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')


def load_trade_journal(filepath: str = "logs/trader_journal.csv") -> pd.DataFrame:
    """
    Load trade journal data and extract unique symbol-date pairs.
    
    Args:
        filepath: Path to trade journal CSV file
        
    Returns:
        DataFrame with unique (symbol, date) pairs
    """
    try:
        if not os.path.exists(filepath):
            print(f"❌ Trade journal file not found: {filepath}")
            print("Please ensure you have trade data in the logs directory.")
            return pd.DataFrame()
        
        # Load trade data
        trades_df = pd.read_csv(filepath)
        
        # Validate required columns
        required_cols = ['date', 'symbol']
        missing_cols = [col for col in required_cols if col not in trades_df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Convert date column to datetime
        trades_df['date'] = pd.to_datetime(trades_df['date'])
        
        # Extract unique symbol-date pairs
        unique_pairs = trades_df[['symbol', 'date']].drop_duplicates()
        unique_pairs = unique_pairs.sort_values(['symbol', 'date'])
        
        print(f"✓ Loaded {len(unique_pairs)} unique symbol-date pairs from {filepath}")
        return unique_pairs
        
    except Exception as e:
        print(f"❌ Error loading trade journal: {e}")
        return pd.DataFrame()


def download_historical_data(symbol: str, target_date: datetime, days_back: int = 90) -> Optional[pd.DataFrame]:
    """
    Download historical price data for a symbol up to a target date.
    
    Args:
        symbol: Trading symbol
        target_date: Date up to which to download data
        days_back: Number of days of historical data to download
        
    Returns:
        DataFrame with price data or None if failed
    """
    try:
        # Calculate start date
        start_date = target_date - timedelta(days=days_back)
        
        # Download data using yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=target_date + timedelta(days=1))
        
        if data.empty:
            print(f"⚠ No data available for {symbol} from {start_date.date()} to {target_date.date()}")
            return None
        
        # Ensure we have sufficient data
        if len(data) < 50:
            print(f"⚠ Insufficient data for {symbol}: {len(data)} observations (need >=50)")
            return None
        
        print(f"✓ Downloaded {len(data)} days of data for {symbol} up to {target_date.date()}")
        return data
        
    except Exception as e:
        print(f"⚠ Error downloading data for {symbol}: {e}")
        return None


def calculate_returns_and_fit_garch(data: pd.DataFrame) -> Optional[Tuple[float, str]]:
    """
    Calculate returns and fit GARCH model to forecast volatility.
    
    Args:
        data: DataFrame with price data
        
    Returns:
        Tuple of (forecasted_volatility, regime) or None if failed
    """
    try:
        # Calculate daily returns
        prices = data['Close'].dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        
        if len(returns) < 50:
            return None
        
        # Check for extreme values
        if returns.std() == 0 or np.isinf(returns).any() or np.isnan(returns).any():
            return None
        
        # Scale returns to percentage for better numerical stability
        scaled_returns = returns * 100
        
        # Fit GARCH(1,1) model
        model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, mean='Constant')
        fitted_model = model.fit(disp='off')
        
        # Generate 1-step ahead forecast
        forecast = fitted_model.forecast(horizon=1)
        variance_forecast = forecast.variance.iloc[-1].values[0]
        volatility_forecast = np.sqrt(variance_forecast) / 100  # Convert back from percentage
        
        # Classify regime
        if volatility_forecast > 0.025:
            regime = "HIGH"
        elif volatility_forecast < 0.012:
            regime = "LOW"
        else:
            regime = "NORMAL"
        
        return volatility_forecast, regime
        
    except Exception as e:
        print(f"⚠ GARCH model failed: {e}")
        return None


def save_forecasts_to_csv(forecasts: List[dict], filepath: str = "logs/garch_forecast_log.csv") -> bool:
    """
    Save forecast results to CSV file.
    
    Args:
        forecasts: List of forecast dictionaries
        filepath: Output CSV file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure logs directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Create DataFrame
        forecasts_df = pd.DataFrame(forecasts)
        
        # Check if file exists to determine if we need headers
        file_exists = os.path.exists(filepath)
        
        # Prepare data for CSV
        csv_data = []
        for forecast in forecasts:
            csv_data.append([
                forecast['date'].strftime('%Y-%m-%d %H:%M:%S'),
                forecast['symbol'],
                forecast['period'],
                f"{forecast['forecast_vol']:.6f}",
                forecast['regime']
            ])
        
        # Write to CSV
        with open(filepath, 'a', newline='', encoding='utf-8') as csvfile:
            import csv
            writer = csv.writer(csvfile)
            
            # Write headers if file is new
            if not file_exists:
                headers = ['date', 'symbol', 'period', 'forecast_vol', 'regime']
                writer.writerow(headers)
            
            # Write data rows
            for row in csv_data:
                writer.writerow(row)
        
        print(f"✓ Saved {len(forecasts)} forecasts to {filepath}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving forecasts: {e}")
        return False


def generate_historical_forecasts() -> None:
    """
    Main function to generate historical GARCH forecasts from trade journal.
    """
    print("Starting Historical GARCH Forecast Generation...")
    print("=" * 60)
    
    # Load trade journal
    unique_pairs = load_trade_journal()
    
    if unique_pairs.empty:
        print("❌ No trade data available. Exiting.")
        return
    
    # Track progress
    total_pairs = len(unique_pairs)
    successful_forecasts = 0
    failed_forecasts = 0
    forecasts_list = []
    
    print(f"Processing {total_pairs} symbol-date pairs...")
    print("-" * 60)
    
    # Symbol mapping for proxy forecasts
    symbol_mapping = {
        "MGC1": "GLD",
        "MES": "SPY", 
        "MYM": "DIA"
    }
    
    # Process each unique symbol-date pair
    for idx, (_, row) in enumerate(unique_pairs.iterrows(), 1):
        symbol = row['symbol']
        target_date = row['date']
        
        # Translate symbol for yfinance download
        yf_symbol = symbol_mapping.get(symbol, symbol)
        
        print(f"[{idx}/{total_pairs}] Processing {symbol} for {target_date.date()}")
        if symbol != yf_symbol:
            print(f"  🔄 Using proxy: {symbol} → {yf_symbol}")
        
        # Download historical data using translated symbol
        data = download_historical_data(yf_symbol, target_date)
        
        if data is None:
            failed_forecasts += 1
            continue
        
        # Fit GARCH model and forecast
        result = calculate_returns_and_fit_garch(data)
        
        if result is None:
            failed_forecasts += 1
            continue
        
        forecast_vol, regime = result
        
        # Store forecast
        forecast_data = {
            'date': target_date,
            'symbol': symbol,
            'period': 'historical',
            'forecast_vol': forecast_vol,
            'regime': regime
        }
        
        forecasts_list.append(forecast_data)
        successful_forecasts += 1
        
        print(f"  ✓ {symbol}: {forecast_vol:.4f} ({forecast_vol:.2%}) - {regime}")
    
    # Save results
    if forecasts_list:
        if save_forecasts_to_csv(forecasts_list):
            print("\n" + "=" * 60)
            print("FORECAST GENERATION COMPLETE")
            print("=" * 60)
            print(f"Total pairs processed: {total_pairs}")
            print(f"Successful forecasts: {successful_forecasts}")
            print(f"Failed forecasts: {failed_forecasts}")
            print(f"Success rate: {(successful_forecasts/total_pairs)*100:.1f}%")
            
            # Show regime distribution
            regimes = [f['regime'] for f in forecasts_list]
            regime_counts = pd.Series(regimes).value_counts()
            print(f"\nRegime Distribution:")
            for regime, count in regime_counts.items():
                pct = (count / len(forecasts_list)) * 100
                print(f"  {regime}: {count} forecasts ({pct:.1f}%)")
            
            print("\n✅ Forecasts generated and saved.")
        else:
            print("❌ Failed to save forecasts.")
    else:
        print("❌ No successful forecasts generated.")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate historical GARCH forecasts from trade journal")
    parser.add_argument('--trades', default='logs/trader_journal.csv', help='Path to trade journal CSV')
    parser.add_argument('--output', default='logs/garch_forecast_log.csv', help='Output forecast CSV path')
    parser.add_argument('--days-back', type=int, default=90, help='Days of historical data to download')
    
    args = parser.parse_args()
    
    try:
        # Update the function to use command line arguments
        print("Starting Historical GARCH Forecast Generation...")
        print("=" * 60)
        
        # Load trade journal with custom path
        unique_pairs = load_trade_journal(args.trades)
        
        if unique_pairs.empty:
            print("❌ No trade data available. Exiting.")
            return
        
        # Track progress
        total_pairs = len(unique_pairs)
        successful_forecasts = 0
        failed_forecasts = 0
        forecasts_list = []
        
        print(f"Processing {total_pairs} symbol-date pairs...")
        print("-" * 60)
        
        # Symbol mapping for proxy forecasts
        symbol_mapping = {
            "MGC1": "GLD",
            "MES": "SPY", 
            "MYM": "DIA"
        }
        
        # Process each unique symbol-date pair
        for idx, (_, row) in enumerate(unique_pairs.iterrows(), 1):
            symbol = row['symbol']
            target_date = row['date']
            
            # Translate symbol for yfinance download
            yf_symbol = symbol_mapping.get(symbol, symbol)
            
            print(f"[{idx}/{total_pairs}] Processing {symbol} for {target_date.date()}")
            if symbol != yf_symbol:
                print(f"  🔄 Using proxy: {symbol} → {yf_symbol}")
            
            # Download historical data using translated symbol
            data = download_historical_data(yf_symbol, target_date, args.days_back)
            
            if data is None:
                failed_forecasts += 1
                continue
            
            # Fit GARCH model and forecast
            result = calculate_returns_and_fit_garch(data)
            
            if result is None:
                failed_forecasts += 1
                continue
            
            forecast_vol, regime = result
            
            # Store forecast
            forecast_data = {
                'date': target_date,
                'symbol': symbol,
                'period': 'historical',
                'forecast_vol': forecast_vol,
                'regime': regime
            }
            
            forecasts_list.append(forecast_data)
            successful_forecasts += 1
            
            print(f"  ✓ {symbol}: {forecast_vol:.4f} ({forecast_vol:.2%}) - {regime}")
        
        # Save results with custom output path
        if forecasts_list:
            if save_forecasts_to_csv(forecasts_list, args.output):
                print("\n" + "=" * 60)
                print("FORECAST GENERATION COMPLETE")
                print("=" * 60)
                print(f"Total pairs processed: {total_pairs}")
                print(f"Successful forecasts: {successful_forecasts}")
                print(f"Failed forecasts: {failed_forecasts}")
                print(f"Success rate: {(successful_forecasts/total_pairs)*100:.1f}%")
                
                # Show regime distribution
                regimes = [f['regime'] for f in forecasts_list]
                regime_counts = pd.Series(regimes).value_counts()
                print(f"\nRegime Distribution:")
                for regime, count in regime_counts.items():
                    pct = (count / len(forecasts_list)) * 100
                    print(f"  {regime}: {count} forecasts ({pct:.1f}%)")
                
                print("\n✅ Forecasts generated and saved.")
            else:
                print("❌ Failed to save forecasts.")
        else:
            print("❌ No successful forecasts generated.")
            
    except KeyboardInterrupt:
        print("\n❌ Forecast generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 