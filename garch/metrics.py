"""
Volatility Metrics and Risk Analysis Module

This module provides comprehensive volatility analysis tools including traditional
risk metrics, regime detection, and advanced volatility statistics for financial
time series analysis and strategy evaluation.

Key Features:
    - Traditional risk metrics (Sharpe, Sortino, Calmar ratios)
    - Maximum drawdown and drawdown duration analysis
    - Volatility regime detection and clustering
    - Value at Risk (VaR) and Expected Shortfall calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import warnings


class VolatilityMetrics:
    """
    Comprehensive volatility and risk metrics calculator.
    
    This class provides methods to calculate various risk-adjusted performance
    metrics and volatility statistics for financial time series.
    """
    
    def __init__(self):
        """Initialize volatility metrics calculator."""
        pass
    
    def calculate_sharpe_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.02,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate annualized Sharpe ratio.
        
        Args:
            returns: Time series of returns
            risk_free_rate: Risk-free rate (annualized)
            annualization_factor: Days per year for annualization
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / annualization_factor)
        return (excess_returns.mean() * annualization_factor) / (returns.std() * np.sqrt(annualization_factor))
    
    def calculate_sortino_ratio(
        self, 
        returns: pd.Series, 
        risk_free_rate: float = 0.02,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate annualized Sortino ratio (downside deviation).
        
        Args:
            returns: Time series of returns
            risk_free_rate: Risk-free rate (annualized)
            annualization_factor: Days per year for annualization
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / annualization_factor)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return np.inf
            
        downside_deviation = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(annualization_factor)
        
        if downside_deviation == 0:
            return 0.0
            
        return (excess_returns.mean() * annualization_factor) / downside_deviation
    
    def calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown from equity curve.
        
        Args:
            equity_curve: Time series of equity values
            
        Returns:
            Maximum drawdown as positive fraction
        """
        if len(equity_curve) == 0:
            return 0.0
            
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max
        
        return abs(drawdown.min())
    
    def calculate_calmar_ratio(
        self, 
        returns: pd.Series, 
        equity_curve: pd.Series,
        annualization_factor: int = 252
    ) -> float:
        """
        Calculate Calmar ratio (annualized return / max drawdown).
        
        Args:
            returns: Time series of returns
            equity_curve: Time series of equity values
            annualization_factor: Days per year for annualization
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0 or len(equity_curve) == 0:
            return 0.0
            
        annualized_return = returns.mean() * annualization_factor
        max_dd = self.calculate_max_drawdown(equity_curve)
        
        if max_dd == 0:
            return np.inf if annualized_return > 0 else 0.0
            
        return annualized_return / max_dd
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Time series of returns
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            VaR value (positive number representing loss)
        """
        if len(returns) == 0:
            return 0.0
            
        return -np.percentile(returns, confidence_level * 100)
    
    def calculate_expected_shortfall(
        self, 
        returns: pd.Series, 
        confidence_level: float = 0.05
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Time series of returns
            confidence_level: Confidence level for ES calculation
            
        Returns:
            Expected shortfall (positive number representing expected loss)
        """
        if len(returns) == 0:
            return 0.0
            
        var_threshold = np.percentile(returns, confidence_level * 100)
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
            
        return -tail_losses.mean()
    
    def detect_volatility_regimes(
        self, 
        returns: pd.Series, 
        window: int = 21,
        threshold_multiplier: float = 1.5
    ) -> pd.Series:
        """
        Detect volatility regimes using rolling volatility.
        
        Args:
            returns: Time series of returns
            window: Rolling window for volatility calculation
            threshold_multiplier: Multiplier for regime threshold
            
        Returns:
            Series indicating regime (0=low, 1=high volatility)
        """
        if len(returns) < window:
            return pd.Series(index=returns.index, data=0)
            
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=window).std()
        
        # Calculate threshold based on historical volatility
        vol_mean = rolling_vol.mean()
        vol_std = rolling_vol.std()
        threshold = vol_mean + threshold_multiplier * vol_std
        
        # Classify regimes
        regimes = (rolling_vol > threshold).astype(int)
        
        return regimes
    
    def calculate_drawdown_duration(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Calculate drawdown duration statistics.
        
        Args:
            equity_curve: Time series of equity values
            
        Returns:
            Dictionary with drawdown duration statistics
        """
        if len(equity_curve) == 0:
            return {'max_duration': 0, 'avg_duration': 0, 'num_drawdowns': 0}
            
        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Identify periods where equity is below running max
        is_drawdown = equity_curve < running_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, is_dd in is_drawdown.items():
            if is_dd and not in_drawdown:
                # Start of new drawdown
                in_drawdown = True
                start_date = date
            elif not is_dd and in_drawdown:
                # End of drawdown
                in_drawdown = False
                if start_date is not None:
                    duration = (date - start_date).days
                    drawdown_periods.append(duration)
        
        # Handle case where series ends in drawdown
        if in_drawdown and start_date is not None:
            duration = (equity_curve.index[-1] - start_date).days
            drawdown_periods.append(duration)
        
        if len(drawdown_periods) == 0:
            return {'max_duration': 0, 'avg_duration': 0, 'num_drawdowns': 0}
        
        return {
            'max_duration': max(drawdown_periods),
            'avg_duration': sum(drawdown_periods) / len(drawdown_periods),
            'num_drawdowns': len(drawdown_periods)
        }
    
    def calculate_performance_summary(
        self, 
        returns: pd.Series, 
        equity_curve: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate comprehensive performance summary.
        
        Args:
            returns: Time series of returns
            equity_curve: Time series of equity values
            risk_free_rate: Risk-free rate for Sharpe calculation
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        summary = {
            'total_return': (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1 if len(equity_curve) > 0 else 0,
            'annualized_return': returns.mean() * 252 if len(returns) > 0 else 0,
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(returns, risk_free_rate),
            'sortino_ratio': self.calculate_sortino_ratio(returns, risk_free_rate),
            'max_drawdown': self.calculate_max_drawdown(equity_curve),
            'calmar_ratio': self.calculate_calmar_ratio(returns, equity_curve),
            'var_95': self.calculate_var(returns, 0.05),
            'expected_shortfall_95': self.calculate_expected_shortfall(returns, 0.05)
        }
        
        # Add drawdown duration stats
        dd_stats = self.calculate_drawdown_duration(equity_curve)
        summary.update({
            'max_drawdown_duration': dd_stats['max_duration'],
            'avg_drawdown_duration': dd_stats['avg_duration'],
            'num_drawdowns': dd_stats['num_drawdowns']
        })
        
        return summary 