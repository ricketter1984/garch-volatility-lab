#!/usr/bin/env python3
"""
CME Futures Data Scraper

This module provides functionality to scrape CME futures data
for GARCH volatility analysis.

Key Features:
    - Fetch CME futures price data
    - Handle different futures contracts
    - Return standardized price series
    - Fallback to yfinance for unavailable contracts
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from typing import Optional, Union
import requests
import time

warnings.filterwarnings('ignore')


def get_price_series(symbol: str, start_date: Optional[datetime] = None, 
                    end_date: Optional[datetime] = None) -> pd.Series:
    """
    Get CME futures price series for a given symbol.
    
    Args:
        symbol: CME futures symbol (e.g., 'ES', 'NQ', 'YM', 'GC')
        start_date: Start date for data (default: 2 years ago)
        end_date: End date for data (default: today)
        
    Returns:
        Series of daily closing prices
    """
    try:
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=730)  # 2 years
        
        print(f"📊 Fetching CME data for {symbol} from {start_date.date()} to {end_date.date()}")
        
        # CME symbol mapping to yfinance symbols
        cme_mapping = {
            'ES': 'ES=F',    # E-mini S&P 500
            'NQ': 'NQ=F',    # E-mini NASDAQ-100
            'YM': 'YM=F',    # E-mini Dow
            'RTY': 'RTY=F',  # E-mini Russell 2000
            'GC': 'GC=F',    # Gold
            'SI': 'SI=F',    # Silver
            'CL': 'CL=F',    # Crude Oil
            'NG': 'NG=F',    # Natural Gas
            'ZC': 'ZC=F',    # Corn
            'ZS': 'ZS=F',    # Soybeans
            'ZW': 'ZW=F',    # Wheat
            '6E': '6E=F',    # Euro FX
            '6J': '6J=F',    # Japanese Yen
            '6B': '6B=F',    # British Pound
            '6A': '6A=F',    # Australian Dollar
            '6C': '6C=F',    # Canadian Dollar
            '6S': '6S=F',    # Swiss Franc
            '6M': '6M=F',    # Mexican Peso
            '6N': '6N=F',    # New Zealand Dollar
            '6R': '6R=F',    # Russian Ruble
            '6Z': '6Z=F',    # South African Rand
            '6L': '6L=F',    # Brazilian Real
            '6K': '6K=F',    # Korean Won
            '6H': '6H=F',    # Hungarian Forint
            '6P': '6P=F',    # Polish Zloty
            '6T': '6T=F',    # Turkish Lira
            '6U': '6U=F',    # Chinese Yuan
            '6V': '6V=F',    # Indian Rupee
            '6W': '6W=F',    # Singapore Dollar
            '6X': '6X=F',    # Swedish Krona
            '6Y': '6Y=F',    # Norwegian Krone
            '6Z': '6Z=F',    # South African Rand
            '6A': '6A=F',    # Australian Dollar
            '6B': '6B=F',    # British Pound
            '6C': '6C=F',    # Canadian Dollar
            '6E': '6E=F',    # Euro FX
            '6J': '6J=F',    # Japanese Yen
            '6M': '6M=F',    # Mexican Peso
            '6N': '6N=F',    # New Zealand Dollar
            '6R': '6R=F',    # Russian Ruble
            '6S': '6S=F',    # Swiss Franc
            '6T': '6T=F',    # Turkish Lira
            '6U': '6U=F',    # Chinese Yuan
            '6V': '6V=F',    # Indian Rupee
            '6W': '6W=F',    # Singapore Dollar
            '6X': '6X=F',    # Swedish Krona
            '6Y': '6Y=F',    # Norwegian Krone
            '6Z': '6Z=F',    # South African Rand
        }
        
        # Get yfinance symbol
        yf_symbol = cme_mapping.get(symbol.upper(), f"{symbol}=F")
        
        # Try to fetch data using yfinance (as proxy for CME data)
        ticker = yf.Ticker(yf_symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"⚠ No data available for {symbol} ({yf_symbol})")
            return pd.Series()
        
        # Extract closing prices
        prices = data['Close']
        
        if len(prices) < 50:
            print(f"⚠ Insufficient data for {symbol}: {len(prices)} observations")
            return pd.Series()
        
        print(f"✓ Loaded {len(prices)} days of CME data for {symbol}")
        return prices
        
    except Exception as e:
        print(f"❌ Error fetching CME data for {symbol}: {e}")
        return pd.Series()


def get_cme_contracts() -> dict:
    """
    Get list of available CME contracts.
    
    Returns:
        Dictionary of contract categories and symbols
    """
    return {
        'Equity Index': ['ES', 'NQ', 'YM', 'RTY'],
        'Metals': ['GC', 'SI', 'PL', 'PA'],
        'Energy': ['CL', 'NG', 'HO', 'RB'],
        'Grains': ['ZC', 'ZS', 'ZW', 'ZC'],
        'Livestock': ['LE', 'HE', 'GF', 'FC'],
        'Softs': ['CT', 'CC', 'KC', 'SB'],
        'Currencies': ['6E', '6J', '6B', '6A', '6C', '6S', '6M', '6N'],
        'Interest Rates': ['GE', 'TU', 'FV', 'TY', 'US'],
    }


def validate_cme_symbol(symbol: str) -> bool:
    """
    Validate if a symbol is a valid CME contract.
    
    Args:
        symbol: Symbol to validate
        
    Returns:
        True if valid CME symbol, False otherwise
    """
    contracts = get_cme_contracts()
    all_symbols = []
    for category in contracts.values():
        all_symbols.extend(category)
    
    return symbol.upper() in all_symbols


if __name__ == "__main__":
    """Test the CME scraper functionality."""
    import argparse
    
    parser = argparse.ArgumentParser(description="CME Futures Data Scraper")
    parser.add_argument('--symbol', default='ES', help='CME futures symbol')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
    if args.end:
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    
    # Get price series
    prices = get_price_series(args.symbol, start_date, end_date)
    
    if not prices.empty:
        print(f"\n📊 Price Summary for {args.symbol}:")
        print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Observations: {len(prices)}")
        print(f"Latest Price: ${prices.iloc[-1]:.2f}")
        print(f"Price Range: ${prices.min():.2f} - ${prices.max():.2f}")
    else:
        print(f"❌ No data available for {args.symbol}") 