"""
GARCH Volatility Forecasting Module

This module provides functionality for fitting GARCH models and forecasting volatility
using the arch library. It supports various GARCH specifications and includes methods
for model evaluation and forecasting.

Key Features:
    - GARCH(1,1) and extended GARCH model fitting
    - Volatility forecasting with confidence intervals
    - Model diagnostics and evaluation metrics
    - Support for multiple time series (MES, MYM, MGC, M6E)
"""

import numpy as np
import pandas as pd
from arch import arch_model
from typing import Optional, Tuple, Dict, Any
import warnings
import yfinance as yf


class GARCHForecaster:
    """
    GARCH volatility forecasting class.
    
    This class provides methods to fit GARCH models to financial time series
    and generate volatility forecasts.
    """
    
    def __init__(self, vol_model: str = 'GARCH', p: int = 1, q: int = 1):
        """
        Initialize GARCH forecaster.
        
        Args:
            vol_model: Volatility model type ('GARCH', 'EGARCH', 'GJR-GARCH')
            p: Number of lagged variance terms
            q: Number of lagged error terms
        """
        self.vol_model = vol_model
        self.p = p
        self.q = q
        self.model = None
        self.fitted_model = None
        self.data = None
        
    def fit(self, returns: pd.Series, mean: str = 'Constant') -> 'GARCHForecaster':
        """
        Fit GARCH model to return series.
        
        Args:
            returns: Time series of returns
            mean: Mean model specification ('Constant', 'Zero', 'AR')
            
        Returns:
            Self for method chaining
        """
        self.data = returns.dropna()
        
        # Scale returns to percentage for better numerical stability
        scaled_returns = self.data * 100
        
        # Create GARCH model
        self.model = arch_model(
            scaled_returns, 
            vol=self.vol_model, 
            p=self.p, 
            q=self.q,
            mean=mean
        )
        
        # Fit model with error handling
        try:
            self.fitted_model = self.model.fit(disp='off')
        except Exception as e:
            warnings.warn(f"GARCH model fitting failed: {e}")
            self.fitted_model = None
            
        return self
    
    def forecast(self, horizon: int = 1) -> Dict[str, Any]:
        """
        Generate volatility forecast.
        
        Args:
            horizon: Forecast horizon in periods
            
        Returns:
            Dictionary containing forecast statistics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before forecasting")
            
        # Generate forecast
        forecast = self.fitted_model.forecast(horizon=horizon)
        
        # Extract forecast statistics
        forecast_variance = forecast.variance.iloc[-1].values
        forecast_volatility = np.sqrt(forecast_variance / 100)  # Convert back from percentage
        
        return {
            'horizon': horizon,
            'volatility_forecast': forecast_volatility,
            'variance_forecast': forecast_variance / 10000,  # Convert to decimal
            'annualized_volatility': forecast_volatility * np.sqrt(252),
            'forecast_date': self.data.index[-1] + pd.Timedelta(days=1)
        }
    
    def get_model_summary(self) -> str:
        """
        Get fitted model summary.
        
        Returns:
            Model summary string
        """
        if self.fitted_model is None:
            return "Model not fitted"
        return str(self.fitted_model.summary())
    
    def get_parameters(self) -> Dict[str, float]:
        """
        Get fitted model parameters.
        
        Returns:
            Dictionary of model parameters
        """
        if self.fitted_model is None:
            return {}
        return self.fitted_model.params.to_dict()
    
    def calculate_volatility_clusters(self, threshold: float = 2.0) -> pd.Series:
        """
        Identify volatility clustering periods.
        
        Args:
            threshold: Standard deviation threshold for high volatility
            
        Returns:
            Boolean series indicating volatility clusters
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted before analysis")
            
        conditional_vol = self.fitted_model.conditional_volatility / 100
        vol_mean = conditional_vol.mean()
        vol_std = conditional_vol.std()
        
        return conditional_vol > (vol_mean + threshold * vol_std)


def forecast_next_day_volatility(symbol: str = "SPY", period: str = "2y") -> float:
    """
    Forecast next-day volatility for a given symbol using GARCH(1,1) model.
    
    This function downloads historical price data, calculates log returns, fits a GARCH(1,1) 
    model, and forecasts the next-day volatility (standard deviation).
    
    Args:
        symbol: Stock/ETF symbol to forecast (default: "SPY")
        period: Period of historical data to use (default: "2y" for 2 years)
                Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    
    Returns:
        Next-day volatility forecast as standard deviation (float), rounded to 5 decimal places.
        Returns 0.0 if unable to generate forecast.
    
    Example:
        >>> vol = forecast_next_day_volatility("SPY")
        >>> print(f"Next-day volatility: {vol:.2%}")
        Next-day volatility: 1.12%
        
        >>> vol = forecast_next_day_volatility("GLD", period="1y")
        >>> print(f"Gold volatility forecast: {vol:.4f}")
        Gold volatility forecast: 0.0089
    """
    try:
        # Download historical data using yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty or len(data) < 50:  # Need sufficient data for GARCH
            warnings.warn(f"Insufficient data for {symbol}. Need at least 50 observations.")
            return 0.0
        
        # Calculate log returns
        prices = data['Close'].dropna()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        
        if len(log_returns) < 50:
            warnings.warn(f"Insufficient return data for {symbol} after cleaning.")
            return 0.0
        
        # Check for extreme values that might cause fitting issues
        if log_returns.std() == 0 or np.isinf(log_returns).any() or np.isnan(log_returns).any():
            warnings.warn(f"Invalid return data for {symbol}.")
            return 0.0
        
        # Scale returns to percentage for better numerical stability
        scaled_returns = log_returns * 100
        
        # Fit GARCH(1,1) model
        model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, mean='Constant')
        fitted_model = model.fit(disp='off')
        
        # Generate next-day forecast
        forecast = fitted_model.forecast(horizon=1)
        
        # Extract variance forecast and convert to standard deviation
        variance_forecast = forecast.variance.iloc[-1].values[0]  # Get the forecast value
        volatility_forecast = np.sqrt(variance_forecast) / 100  # Convert back from percentage
        
        # Round to 5 decimal places
        return round(volatility_forecast, 5)
        
    except Exception as e:
        warnings.warn(f"Error forecasting volatility for {symbol}: {str(e)}")
        return 0.0 