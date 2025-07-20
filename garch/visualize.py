"""
GARCH Volatility Visualization Module

This module provides visualization tools for GARCH overlay analysis results,
including heatmaps and performance charts.

Key Features:
    - Heatmap visualization of strategy performance by volatility regime
    - Performance matrix analysis
    - Interactive plotting capabilities
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional, Tuple
import warnings

# Set style for better visualizations
plt.style.use('default')
sns.set_palette("husl")
warnings.filterwarnings('ignore')


def plot_overlay_heatmap(results_df: pd.DataFrame, 
                        save_path: Optional[str] = "logs/overlay_heatmap.png",
                        figsize: Tuple[int, int] = (10, 8),
                        return_plot: bool = False) -> Optional[plt.Figure]:
    """
    Create a heatmap visualization of strategy performance by volatility regime.
    
    Args:
        results_df: DataFrame with overlay results including columns:
                   setup, volatility_regime, trades, win_rate, avg_r, net_r
        save_path: Optional path to save the heatmap image
        figsize: Figure size as (width, height)
        return_plot: Whether to return the plot object for further chaining
        
    Returns:
        matplotlib Figure object if return_plot=True, None otherwise
    """
    try:
        # Validate required columns (handle different column naming conventions)
        required_cols = ['setup', 'volatility_regime', 'win_rate']
        missing_cols = [col for col in required_cols if col not in results_df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Map column names to expected format
        column_mapping = {
            'total_trades': 'trades',
            'avg_r_multiple': 'avg_r', 
            'total_net_r': 'net_r'
        }
        
        # Create a copy with standardized column names
        df = results_df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure we have the required columns
        if 'trades' not in df.columns and 'total_trades' in df.columns:
            df['trades'] = df['total_trades']
        if 'avg_r' not in df.columns and 'avg_r_multiple' in df.columns:
            df['avg_r'] = df['avg_r_multiple']
        if 'net_r' not in df.columns and 'total_net_r' in df.columns:
            df['net_r'] = df['total_net_r']
        
        # Create pivot table for heatmap
        heatmap_data = df.pivot_table(
            index='setup',
            columns='volatility_regime',
            values='avg_r',
            aggfunc='mean'
        )
        
        # Handle missing values
        heatmap_data = heatmap_data.fillna(np.nan)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Average R-Multiple'},
            ax=ax,
            linewidths=0.5,
            linecolor='white'
        )
        
        # Customize the plot
        ax.set_title('Strategy Performance by Volatility Regime', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Volatility Regime', fontsize=12, fontweight='bold')
        ax.set_ylabel('Setup', fontsize=12, fontweight='bold')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Heatmap saved to: {save_path}")
            except Exception as e:
                print(f"⚠ Could not save heatmap: {e}")
        
        # Show the plot
        plt.show()
        
        # Return plot object if requested
        if return_plot:
            return fig
        else:
            plt.close()
            return None
            
    except Exception as e:
        print(f"❌ Error creating heatmap: {e}")
        return None


def plot_performance_summary(results_df: pd.DataFrame,
                           save_path: Optional[str] = "logs/performance_summary.png",
                           figsize: Tuple[int, int] = (12, 8)) -> Optional[plt.Figure]:
    """
    Create a comprehensive performance summary visualization.
    
    Args:
        results_df: DataFrame with overlay results
        save_path: Optional path to save the plot
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib Figure object
    """
    try:
        # Map column names to expected format
        column_mapping = {
            'total_trades': 'trades',
            'avg_r_multiple': 'avg_r', 
            'total_net_r': 'net_r'
        }
        
        # Create a copy with standardized column names
        df = results_df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure we have the required columns
        if 'trades' not in df.columns and 'total_trades' in df.columns:
            df['trades'] = df['total_trades']
        if 'avg_r' not in df.columns and 'avg_r_multiple' in df.columns:
            df['avg_r'] = df['avg_r_multiple']
        if 'net_r' not in df.columns and 'total_net_r' in df.columns:
            df['net_r'] = df['total_net_r']
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Win Rate by Setup and Regime
        win_rate_pivot = df.pivot_table(
            index='setup', 
            columns='volatility_regime', 
            values='win_rate', 
            aggfunc='mean'
        )
        sns.heatmap(win_rate_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=50, cbar_kws={'label': 'Win Rate (%)'}, ax=ax1)
        ax1.set_title('Win Rate by Setup and Regime')
        
        # 2. Total Net R by Setup and Regime
        net_r_pivot = df.pivot_table(
            index='setup', 
            columns='volatility_regime', 
            values='net_r', 
            aggfunc='sum'
        )
        sns.heatmap(net_r_pivot, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=0, cbar_kws={'label': 'Net R'}, ax=ax2)
        ax2.set_title('Total Net R by Setup and Regime')
        
        # 3. Number of Trades by Setup and Regime
        trades_pivot = df.pivot_table(
            index='setup', 
            columns='volatility_regime', 
            values='trades', 
            aggfunc='sum'
        )
        sns.heatmap(trades_pivot, annot=True, fmt='.0f', cmap='Blues', 
                   cbar_kws={'label': 'Number of Trades'}, ax=ax3)
        ax3.set_title('Number of Trades by Setup and Regime')
        
        # 4. Average R by Setup (bar chart)
        avg_r_by_setup = df.groupby('setup')['avg_r'].mean().sort_values(ascending=True)
        colors = ['green' if x > 0 else 'red' for x in avg_r_by_setup.values]
        avg_r_by_setup.plot(kind='barh', ax=ax4, color=colors)
        ax4.set_title('Average R by Setup')
        ax4.set_xlabel('Average R-Multiple')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Performance summary saved to: {save_path}")
            except Exception as e:
                print(f"⚠ Could not save performance summary: {e}")
        
        plt.show()
        return fig
        
    except Exception as e:
        print(f"❌ Error creating performance summary: {e}")
        return None


def plot_regime_distribution(results_df: pd.DataFrame,
                           save_path: Optional[str] = "logs/regime_distribution.png",
                           figsize: Tuple[int, int] = (10, 6)) -> Optional[plt.Figure]:
    """
    Create a visualization of regime distribution and performance.
    
    Args:
        results_df: DataFrame with overlay results
        save_path: Optional path to save the plot
        figsize: Figure size as (width, height)
        
    Returns:
        matplotlib Figure object
    """
    try:
        # Map column names to expected format
        column_mapping = {
            'total_trades': 'trades',
            'avg_r_multiple': 'avg_r', 
            'total_net_r': 'net_r'
        }
        
        # Create a copy with standardized column names
        df = results_df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df[new_col] = df[old_col]
        
        # Ensure we have the required columns
        if 'trades' not in df.columns and 'total_trades' in df.columns:
            df['trades'] = df['total_trades']
        if 'avg_r' not in df.columns and 'avg_r_multiple' in df.columns:
            df['avg_r'] = df['avg_r_multiple']
        if 'net_r' not in df.columns and 'total_net_r' in df.columns:
            df['net_r'] = df['total_net_r']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # 1. Regime distribution pie chart
        regime_counts = df['volatility_regime'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99']  # Red, Blue, Green
        ax1.pie(regime_counts.values, labels=regime_counts.index, autopct='%1.1f%%', 
               colors=colors[:len(regime_counts)])
        ax1.set_title('Distribution of Volatility Regimes')
        
        # 2. Average performance by regime
        regime_performance = df.groupby('volatility_regime').agg({
            'avg_r': 'mean',
            'win_rate': 'mean',
            'trades': 'sum'
        }).round(2)
        
        x = np.arange(len(regime_performance))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, regime_performance['avg_r'], width, 
                        label='Avg R', color='skyblue')
        bars2 = ax2.bar(x + width/2, regime_performance['win_rate']/100, width, 
                        label='Win Rate', color='lightcoral')
        
        ax2.set_xlabel('Volatility Regime')
        ax2.set_ylabel('Performance Metrics')
        ax2.set_title('Performance by Volatility Regime')
        ax2.set_xticks(x)
        ax2.set_xticklabels(regime_performance.index)
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"✓ Regime distribution saved to: {save_path}")
            except Exception as e:
                print(f"⚠ Could not save regime distribution: {e}")
        
        plt.show()
        return fig
        
    except Exception as e:
        print(f"❌ Error creating regime distribution: {e}")
        return None


if __name__ == "__main__":
    """Test the visualization functions with sample data."""
    try:
        # Try to load the overlay summary data
        if os.path.exists("logs/garch_overlay_summary.csv"):
            print("Loading overlay summary data...")
            df = pd.read_csv("logs/garch_overlay_summary.csv")
            print(f"✓ Loaded {len(df)} rows of overlay data")
            
            # Create visualizations
            print("\nCreating heatmap...")
            plot_overlay_heatmap(df)
            
            print("\nCreating performance summary...")
            plot_performance_summary(df)
            
            print("\nCreating regime distribution...")
            plot_regime_distribution(df)
            
        else:
            print("❌ No overlay summary data found at logs/garch_overlay_summary.csv")
            print("Run the overlay analysis first: python garch/overlay.py")
            
    except Exception as e:
        print(f"❌ Error in visualization test: {e}") 