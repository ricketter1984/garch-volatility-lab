"""
GARCH Strategy Backtesting Module

This module provides functionality for backtesting trading strategies that incorporate
GARCH volatility forecasts. It includes position sizing based on volatility targeting
and strategy performance evaluation with volatility-adjusted metrics.

Key Features:
    - Volatility-adjusted position sizing
    - GARCH-based regime detection for strategy overlay
    - Risk-adjusted performance metrics
    - Strategy comparison with and without GARCH overlay
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
from arch import arch_model
import warnings
from .forecast import GARCHForecaster
from .metrics import VolatilityMetrics


class GARCHBacktester:
    """
    Backtesting engine for strategies incorporating GARCH volatility forecasts.
    
    This class provides methods to backtest trading strategies with volatility
    overlays and position sizing based on GARCH forecasts.
    """
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize GARCH backtester.
        
        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.results = {}
        self.trades = pd.DataFrame()
        self.equity_curve = pd.Series()
        
    def volatility_target_position_size(
        self, 
        forecasted_vol: float, 
        target_vol: float = 0.15, 
        price: float = 1.0,
        capital: Optional[float] = None
    ) -> float:
        """
        Calculate position size based on volatility targeting.
        
        Args:
            forecasted_vol: GARCH forecasted volatility (annualized)
            target_vol: Target portfolio volatility (annualized)
            price: Current asset price
            capital: Available capital (defaults to current equity)
            
        Returns:
            Position size (number of shares/contracts)
        """
        if capital is None:
            capital = self.initial_capital
            
        if forecasted_vol <= 0:
            return 0
            
        # Calculate position size for volatility targeting
        vol_scaling = target_vol / forecasted_vol
        notional_target = capital * vol_scaling
        position_size = notional_target / price
        
        return position_size
    
    def backtest_strategy(
        self,
        price_data: pd.Series,
        signals: pd.Series,
        garch_forecaster: GARCHForecaster,
        target_volatility: float = 0.15,
        use_garch_sizing: bool = True,
        transaction_cost: float = 0.001
    ) -> Dict[str, Any]:
        """
        Backtest a strategy with GARCH volatility overlay.
        
        Args:
            price_data: Time series of asset prices
            signals: Trading signals (-1, 0, 1)
            garch_forecaster: Fitted GARCH forecaster
            target_volatility: Target portfolio volatility
            use_garch_sizing: Whether to use GARCH-based position sizing
            transaction_cost: Transaction cost as fraction of trade value
            
        Returns:
            Dictionary containing backtest results
        """
        # Align data
        common_index = price_data.index.intersection(signals.index)
        prices = price_data.reindex(common_index).fillna(method='ffill')
        sigs = signals.reindex(common_index).fillna(0)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        # Initialize tracking variables
        positions = pd.Series(index=common_index, dtype=float)
        equity = pd.Series(index=common_index, dtype=float)
        trades_list = []
        
        current_position = 0
        current_equity = self.initial_capital
        
        for i, date in enumerate(common_index):
            if i == 0:
                equity.iloc[i] = current_equity
                positions.iloc[i] = 0
                continue
                
            signal = sigs.iloc[i]
            price = prices.iloc[i]
            prev_price = prices.iloc[i-1]
            
            # Update equity based on previous position
            if current_position != 0:
                return_pct = (price - prev_price) / prev_price
                current_equity += current_position * prev_price * return_pct
            
            # Calculate new position based on signal and volatility
            if signal != 0 and use_garch_sizing:
                # Get GARCH volatility forecast
                try:
                    forecast_result = garch_forecaster.forecast(horizon=1)
                    forecasted_vol = forecast_result['annualized_volatility'][0]
                except:
                    forecasted_vol = 0.20  # Default volatility assumption
                    
                target_position = self.volatility_target_position_size(
                    forecasted_vol, target_volatility, price, current_equity
                ) * signal
            elif signal != 0:
                # Fixed position sizing without GARCH
                target_position = (current_equity * 0.95) / price * signal
            else:
                target_position = 0
            
            # Execute trade if position changes
            if target_position != current_position:
                trade_size = target_position - current_position
                trade_cost = abs(trade_size) * price * transaction_cost
                current_equity -= trade_cost
                
                # Record trade
                if trade_size != 0:
                    trades_list.append({
                        'date': date,
                        'price': price,
                        'size': trade_size,
                        'position_after': target_position,
                        'cost': trade_cost,
                        'signal': signal
                    })
                
                current_position = target_position
            
            positions.iloc[i] = current_position
            equity.iloc[i] = current_equity
        
        # Store results
        self.equity_curve = equity
        self.trades = pd.DataFrame(trades_list)
        
        # Calculate performance metrics
        equity_returns = equity.pct_change().dropna()
        metrics = VolatilityMetrics()
        
        self.results = {
            'total_return': (equity.iloc[-1] / equity.iloc[0]) - 1,
            'annualized_return': (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1,
            'volatility': equity_returns.std() * np.sqrt(252),
            'sharpe_ratio': metrics.calculate_sharpe_ratio(equity_returns),
            'max_drawdown': metrics.calculate_max_drawdown(equity),
            'calmar_ratio': metrics.calculate_calmar_ratio(equity_returns, equity),
            'num_trades': len(self.trades),
            'equity_curve': equity,
            'positions': positions,
            'trades': self.trades
        }
        
        return self.results
    
    def compare_strategies(
        self,
        price_data: pd.Series,
        signals: pd.Series,
        garch_forecaster: GARCHForecaster,
        target_volatility: float = 0.15
    ) -> Dict[str, Any]:
        """
        Compare strategy performance with and without GARCH overlay.
        
        Args:
            price_data: Time series of asset prices
            signals: Trading signals
            garch_forecaster: Fitted GARCH forecaster
            target_volatility: Target portfolio volatility
            
        Returns:
            Dictionary comparing both strategies
        """
        # Backtest with GARCH sizing
        garch_results = self.backtest_strategy(
            price_data, signals, garch_forecaster, 
            target_volatility, use_garch_sizing=True
        )
        
        # Backtest without GARCH sizing
        fixed_results = self.backtest_strategy(
            price_data, signals, garch_forecaster, 
            target_volatility, use_garch_sizing=False
        )
        
        comparison = {
            'garch_strategy': {
                'total_return': garch_results['total_return'],
                'sharpe_ratio': garch_results['sharpe_ratio'],
                'max_drawdown': garch_results['max_drawdown'],
                'volatility': garch_results['volatility']
            },
            'fixed_strategy': {
                'total_return': fixed_results['total_return'],
                'sharpe_ratio': fixed_results['sharpe_ratio'],
                'max_drawdown': fixed_results['max_drawdown'],
                'volatility': fixed_results['volatility']
            }
        }
        
        # Calculate improvement metrics
        comparison['improvement'] = {
            'return_diff': garch_results['total_return'] - fixed_results['total_return'],
            'sharpe_diff': garch_results['sharpe_ratio'] - fixed_results['sharpe_ratio'],
            'drawdown_improvement': fixed_results['max_drawdown'] - garch_results['max_drawdown']
        }
        
        return comparison


def simulate_garch_volatility_strategy(data: pd.DataFrame, entry_signal_col: str) -> pd.DataFrame:
    """
    Simulate a GARCH volatility-based trading strategy.
    
    This function takes price data and trading signals, fits a GARCH(1,1) model to returns,
    forecasts volatility, and calculates position-sized PnL based on inverse volatility scaling.
    
    Args:
        data: DataFrame with 'Close' column and trading signal column
        entry_signal_col: Name of the column containing trading signals (-1, 0, 1)
    
    Returns:
        DataFrame with original data plus:
        - forecast_vol: GARCH forecasted volatility (daily)
        - position_size: Position size based on inverse volatility (clipped 0.01-0.05)
        - pnl: Daily profit/loss (signal * return * position_size)
        - cumulative: Cumulative PnL
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Close': [100, 101, 99, 102, 98],
        ...     'signal': [1, 1, -1, 1, -1]
        ... })
        >>> result = simulate_garch_volatility_strategy(df, 'signal')
        >>> print(result[['forecast_vol', 'position_size', 'pnl', 'cumulative']])
    """
    # Create a copy of the input data
    df = data.copy()
    
    # Calculate returns
    df['returns'] = df['Close'].pct_change()
    
    # Remove NaN values and ensure we have sufficient data
    df = df.dropna()
    
    if len(df) < 50:
        warnings.warn("Insufficient data for GARCH model. Need at least 50 observations.")
        # Return DataFrame with zero forecasts
        df['forecast_vol'] = 0.01
        df['position_size'] = 0.01
        df['pnl'] = 0.0
        df['cumulative'] = 0.0
        return df
    
    # Initialize forecast columns
    df['forecast_vol'] = np.nan
    df['position_size'] = np.nan
    df['pnl'] = 0.0
    
    # Check if signal column exists
    if entry_signal_col not in df.columns:
        raise ValueError(f"Signal column '{entry_signal_col}' not found in data")
    
    # Minimum window for GARCH fitting
    min_window = 50
    
    # Rolling GARCH forecast
    for i in range(min_window, len(df)):
        try:
            # Get historical returns for fitting (up to current point)
            hist_returns = df['returns'].iloc[:i]
            
            # Scale returns to percentage for better numerical stability
            scaled_returns = hist_returns * 100
            
            # Fit GARCH(1,1) model
            model = arch_model(scaled_returns, vol='GARCH', p=1, q=1, mean='Constant')
            fitted_model = model.fit(disp='off')
            
            # Generate 1-step ahead forecast
            forecast = fitted_model.forecast(horizon=1)
            variance_forecast = forecast.variance.iloc[-1].values[0]
            volatility_forecast = np.sqrt(variance_forecast) / 100  # Convert back from percentage
            
            # Store forecast
            df.iloc[i, df.columns.get_loc('forecast_vol')] = volatility_forecast
            
        except Exception as e:
            # If GARCH fitting fails, use rolling standard deviation as fallback
            window_returns = df['returns'].iloc[max(0, i-21):i]  # 21-day window
            volatility_forecast = window_returns.std()
            df.iloc[i, df.columns.get_loc('forecast_vol')] = volatility_forecast
    
    # Fill initial NaN values with mean of calculated forecasts
    mean_vol = df['forecast_vol'].mean()
    df['forecast_vol'] = df['forecast_vol'].fillna(mean_vol)
    
    # Calculate position size = 1 / forecasted_volatility (clipped between 0.01 and 0.05)
    df['position_size'] = 1 / df['forecast_vol']
    df['position_size'] = df['position_size'].clip(lower=0.01, upper=0.05)
    
    # Calculate daily PnL = signal * return * position_size
    df['pnl'] = df[entry_signal_col] * df['returns'] * df['position_size']
    
    # Calculate cumulative PnL
    df['cumulative'] = df['pnl'].cumsum()
    
    return df


def plot_strategy(df: pd.DataFrame, title: str = "GARCH Volatility Strategy Performance") -> None:
    """
    Plot cumulative PnL from the GARCH volatility strategy backtest.
    
    This function creates a comprehensive plot showing the strategy performance
    including cumulative PnL, volatility forecast, and position sizing.
    
    Args:
        df: DataFrame returned from simulate_garch_volatility_strategy()
        title: Title for the plot (default: "GARCH Volatility Strategy Performance")
    
    Example:
        >>> df = simulate_garch_volatility_strategy(data, 'signal')
        >>> plot_strategy(df, "My Strategy Results")
    """
    # Check if required columns exist
    required_cols = ['cumulative', 'forecast_vol', 'position_size', 'pnl']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. "
                        "DataFrame should be output from simulate_garch_volatility_strategy()")
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Plot 1: Cumulative PnL
    axes[0].plot(df.index, df['cumulative'], linewidth=2, color='blue', alpha=0.8)
    axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
    axes[0].set_title('Cumulative PnL', fontweight='bold')
    axes[0].set_ylabel('Cumulative Return')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(['Cumulative PnL', 'Breakeven'])
    
    # Add performance statistics as text
    total_return = df['cumulative'].iloc[-1]
    max_drawdown = (df['cumulative'] - df['cumulative'].expanding().max()).min()
    win_rate = (df['pnl'] > 0).mean()
    
    stats_text = f'Total Return: {total_return:.4f}\nMax Drawdown: {max_drawdown:.4f}\nWin Rate: {win_rate:.1%}'
    bbox_props = {'boxstyle': 'round', 'facecolor': 'wheat', 'alpha': 0.8}
    axes[0].text(0.02, 0.98, stats_text, transform=axes[0].transAxes, 
                verticalalignment='top', bbox=bbox_props)
    
    # Plot 2: Forecasted Volatility
    axes[1].plot(df.index, df['forecast_vol'], linewidth=1.5, color='orange', alpha=0.8)
    axes[1].axhline(y=df['forecast_vol'].mean(), color='red', linestyle='--', alpha=0.7)
    axes[1].set_title('GARCH Forecasted Volatility', fontweight='bold')
    axes[1].set_ylabel('Daily Volatility')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(['Forecasted Vol', f'Mean: {df["forecast_vol"].mean():.4f}'])
    
    # Plot 3: Position Size
    axes[2].plot(df.index, df['position_size'], linewidth=1.5, color='green', alpha=0.8)
    axes[2].axhline(y=df['position_size'].mean(), color='red', linestyle='--', alpha=0.7)
    axes[2].set_title('Position Size (Inverse Volatility)', fontweight='bold')
    axes[2].set_ylabel('Position Size')
    axes[2].set_xlabel('Time')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(['Position Size', f'Mean: {df["position_size"].mean():.4f}'])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Total Observations: {len(df)}")
    print(f"Total Return: {total_return:.4f}")
    print(f"Annualized Return: {(1 + total_return) ** (252/len(df)) - 1:.2%}")
    print(f"Volatility (PnL): {df['pnl'].std() * np.sqrt(252):.2%}")
    print(f"Sharpe Ratio: {df['pnl'].mean() / df['pnl'].std() * np.sqrt(252):.2f}")
    print(f"Max Drawdown: {max_drawdown:.4f}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Win: {df[df['pnl'] > 0]['pnl'].mean():.6f}")
    print(f"Average Loss: {df[df['pnl'] < 0]['pnl'].mean():.6f}")
    print(f"Profit Factor: {df[df['pnl'] > 0]['pnl'].sum() / abs(df[df['pnl'] < 0]['pnl'].sum()):.2f}")
    print("="*60) 