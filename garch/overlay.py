"""
GARCH Volatility Overlay Analysis Module

This module analyzes trading performance by combining historical trade data
with GARCH volatility forecasts to understand how different setups perform
under various volatility regimes.

Key Features:
    - Merges trade journal data with GARCH forecast data
    - Analyzes performance by setup and volatility regime
    - Computes win rates, R-multiples, and performance metrics
    - Generates summary reports for strategy optimization
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings


class GARCHOverlayAnalyzer:
    """
    Analyzer for combining trade data with GARCH volatility forecasts.
    
    This class loads trade journal data and GARCH forecast data, merges them,
    and provides analysis of trading performance by setup and volatility regime.
    """
    
    def __init__(self):
        """Initialize the overlay analyzer."""
        self.trades_df = pd.DataFrame()
        self.garch_df = pd.DataFrame()
        self.merged_df = pd.DataFrame()
        self.summary_stats = {}
        
    def load_trade_data(self, filepath: str = "logs/trader_journal.csv") -> bool:
        """
        Load historical trade data from CSV file.
        
        Args:
            filepath: Path to the trade journal CSV file
            
        Returns:
            True if successful, False otherwise
            
        Expected columns: date, symbol, setup, direction, entry_price, exit_price, result, r_multiple, notes
        """
        try:
            # Confirm which file is being loaded
            print(f"📄 Using trade journal: {filepath}")
            
            # Check if file exists before attempting to load
            if not os.path.exists(filepath):
                print(f"❌ ERROR: Trade file not found at {filepath}")
                sys.exit(1)
            
            # Load trade data
            self.trades_df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['date', 'symbol', 'setup', 'direction', 'entry_price', 
                           'exit_price', 'result', 'r_multiple', 'notes']
            missing_cols = [col for col in required_cols if col not in self.trades_df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns in trade data: {missing_cols}")
            
            # Convert date column to datetime
            self.trades_df['date'] = pd.to_datetime(self.trades_df['date'])
            
            # Clean and validate data
            self.trades_df = self.trades_df.dropna(subset=['date', 'symbol', 'setup', 'result'])
            
            # Ensure r_multiple is numeric
            self.trades_df['r_multiple'] = pd.to_numeric(self.trades_df['r_multiple'], errors='coerce')
            
            print(f"✓ Loaded {len(self.trades_df)} trades from {filepath}")
            
            # Debug block to verify file reading
            print("\n🔍 DEBUG: Reading trader journal...")
            with open(filepath, "r") as f:
                lines = f.readlines()
                print(f"Total lines in file: {len(lines)}")
                print(f"First 2 rows:\n{lines[:2]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading trade data: {e}")
            return False
    
    def load_garch_data(self, filepath: str = "logs/garch_forecast_log.csv") -> bool:
        """
        Load GARCH forecast data from CSV file.
        
        Args:
            filepath: Path to the GARCH forecast CSV file
            
        Returns:
            True if successful, False otherwise
            
        Expected columns: date, symbol, period, forecast_vol, regime
        """
        try:
            if not os.path.exists(filepath):
                print(f"⚠ GARCH forecast file not found: {filepath}")
                print("Run some GARCH forecasts first using: python run_garch_forecast.py")
                return False
            
            # Load GARCH data
            self.garch_df = pd.read_csv(filepath)
            
            # Validate required columns
            required_cols = ['date', 'symbol', 'period', 'forecast_vol', 'regime']
            missing_cols = [col for col in required_cols if col not in self.garch_df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns in GARCH data: {missing_cols}")
            
            # Convert date column to datetime
            self.garch_df['date'] = pd.to_datetime(self.garch_df['date'])
            
            # Clean data
            self.garch_df = self.garch_df.dropna(subset=['date', 'symbol', 'regime'])
            
            print(f"✓ Loaded {len(self.garch_df)} GARCH forecasts from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading GARCH data: {e}")
            return False
    
    def merge_datasets(self, date_tolerance_days: int = 1) -> bool:
        """
        Merge trade data with GARCH forecast data.
        
        Args:
            date_tolerance_days: Maximum days difference for matching trades to forecasts
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.trades_df.empty:
                print("❌ No trade data loaded")
                return False
            
            if self.garch_df.empty:
                print("❌ No GARCH forecast data loaded")
                return False
            
            # Start with trades dataframe
            self.merged_df = self.trades_df.copy()
            self.merged_df['volatility_regime'] = 'UNKNOWN'
            self.merged_df['forecast_vol'] = np.nan
            
            # Merge with tolerance for date matching
            matches_found = 0
            
            for idx, trade in self.merged_df.iterrows():
                trade_date = trade['date']
                trade_symbol = trade['symbol']
                
                # Find GARCH forecasts within date tolerance for the same symbol
                date_min = trade_date - timedelta(days=date_tolerance_days)
                date_max = trade_date + timedelta(days=date_tolerance_days)
                
                matching_forecasts = self.garch_df[
                    (self.garch_df['symbol'] == trade_symbol) &
                    (self.garch_df['date'] >= date_min) &
                    (self.garch_df['date'] <= date_max)
                ]
                
                if not matching_forecasts.empty:
                    # Use the closest forecast by date
                    closest_forecast = matching_forecasts.loc[
                        (matching_forecasts['date'] - trade_date).abs().idxmin()
                    ]
                    
                    self.merged_df.at[idx, 'volatility_regime'] = closest_forecast['regime']
                    self.merged_df.at[idx, 'forecast_vol'] = closest_forecast['forecast_vol']
                    matches_found += 1
            
            # Remove trades without GARCH matches for analysis
            trades_with_garch = self.merged_df[self.merged_df['volatility_regime'] != 'UNKNOWN']
            trades_without_garch = len(self.merged_df) - len(trades_with_garch)
            
            self.merged_df = trades_with_garch
            
            print(f"✓ Merged datasets: {len(self.merged_df)} trades with GARCH data")
            if trades_without_garch > 0:
                print(f"⚠ {trades_without_garch} trades excluded (no matching GARCH forecast)")
            
            return len(self.merged_df) > 0
            
        except Exception as e:
            print(f"❌ Error merging datasets: {e}")
            return False
    
    def analyze_performance(self) -> Dict[str, Any]:
        """
        Analyze trading performance by setup and volatility regime.
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            if self.merged_df.empty:
                print("❌ No merged data available for analysis")
                return {}
            
            # Group by setup and volatility regime
            grouped = self.merged_df.groupby(['setup', 'volatility_regime'])
            
            # Calculate metrics for each group
            analysis_results = []
            
            for (setup, regime), group in grouped:
                total_trades = len(group)
                wins = len(group[group['result'] == 'win'])
                losses = len(group[group['result'] == 'loss'])
                win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
                
                # R-multiple statistics
                r_multiples = group['r_multiple'].dropna()
                avg_r = r_multiples.mean() if len(r_multiples) > 0 else 0
                total_r = r_multiples.sum() if len(r_multiples) > 0 else 0
                
                # Setup tier classification (assuming tier info in setup name or notes)
                tier_1_count = len(group[group['setup'].str.contains('Tier1|T1', case=False, na=False)])
                tier_2_count = len(group[group['setup'].str.contains('Tier2|T2', case=False, na=False)])
                
                # If no tier info in setup name, check notes
                if tier_1_count == 0 and tier_2_count == 0:
                    tier_1_count = len(group[group['notes'].str.contains('Tier1|T1', case=False, na=False)])
                    tier_2_count = len(group[group['notes'].str.contains('Tier2|T2', case=False, na=False)])
                
                tier_1_pct = (tier_1_count / total_trades) * 100 if total_trades > 0 else 0
                tier_2_pct = (tier_2_count / total_trades) * 100 if total_trades > 0 else 0
                
                # Average volatility for this group
                avg_volatility = group['forecast_vol'].mean() if 'forecast_vol' in group.columns else 0
                
                analysis_results.append({
                    'setup': setup,
                    'volatility_regime': regime,
                    'total_trades': total_trades,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'avg_r_multiple': avg_r,
                    'total_net_r': total_r,
                    'tier_1_pct': tier_1_pct,
                    'tier_2_pct': tier_2_pct,
                    'avg_forecast_vol': avg_volatility
                })
            
            # Convert to DataFrame for easier analysis
            self.summary_stats = pd.DataFrame(analysis_results)
            
            # Overall statistics
            overall_stats = {
                'total_trades_analyzed': len(self.merged_df),
                'unique_setups': self.merged_df['setup'].nunique(),
                'regime_distribution': self.merged_df['volatility_regime'].value_counts().to_dict(),
                'overall_win_rate': (len(self.merged_df[self.merged_df['result'] == 'win']) / len(self.merged_df)) * 100,
                'overall_avg_r': self.merged_df['r_multiple'].mean(),
                'date_range': f"{self.merged_df['date'].min().date()} to {self.merged_df['date'].max().date()}"
            }
            
            return {
                'summary_table': self.summary_stats,
                'overall_stats': overall_stats,
                'detailed_data': self.merged_df
            }
            
        except Exception as e:
            print(f"❌ Error in performance analysis: {e}")
            return {}
    
    def print_summary_report(self, results: Dict[str, Any]) -> None:
        """
        Print a formatted summary report to console.
        
        Args:
            results: Analysis results from analyze_performance()
        """
        try:
            if not results:
                print("❌ No results to display")
                return
            
            print("\n" + "="*80)
            print("GARCH VOLATILITY OVERLAY ANALYSIS REPORT")
            print("="*80)
            
            # Overall statistics
            overall = results['overall_stats']
            print(f"Analysis Period:     {overall['date_range']}")
            print(f"Total Trades:        {overall['total_trades_analyzed']}")
            print(f"Unique Setups:       {overall['unique_setups']}")
            print(f"Overall Win Rate:    {overall['overall_win_rate']:.1f}%")
            print(f"Overall Avg R:       {overall['overall_avg_r']:.2f}")
            
            print(f"\nVolatility Regime Distribution:")
            for regime, count in overall['regime_distribution'].items():
                pct = (count / overall['total_trades_analyzed']) * 100
                print(f"  {regime}: {count} trades ({pct:.1f}%)")
            
            # Summary table
            print(f"\n{'-'*80}")
            print("PERFORMANCE BY SETUP AND VOLATILITY REGIME")
            print(f"{'-'*80}")
            
            if not self.summary_stats.empty:
                # Format and display summary table
                display_df = self.summary_stats.copy()
                display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.1f}%")
                display_df['avg_r_multiple'] = display_df['avg_r_multiple'].apply(lambda x: f"{x:.2f}")
                display_df['total_net_r'] = display_df['total_net_r'].apply(lambda x: f"{x:.1f}R")
                display_df['tier_1_pct'] = display_df['tier_1_pct'].apply(lambda x: f"{x:.0f}%")
                display_df['tier_2_pct'] = display_df['tier_2_pct'].apply(lambda x: f"{x:.0f}%")
                display_df['avg_forecast_vol'] = display_df['avg_forecast_vol'].apply(lambda x: f"{x:.3f}")
                
                # Rename columns for display
                display_df = display_df.rename(columns={
                    'setup': 'Setup',
                    'volatility_regime': 'Vol Regime',
                    'total_trades': 'Trades',
                    'wins': 'Wins',
                    'losses': 'Losses',
                    'win_rate': 'Win Rate',
                    'avg_r_multiple': 'Avg R',
                    'total_net_r': 'Net R',
                    'tier_1_pct': 'T1%',
                    'tier_2_pct': 'T2%',
                    'avg_forecast_vol': 'Avg Vol'
                })
                
                print(display_df.to_string(index=False))
            
            # Key insights
            print(f"\n{'-'*80}")
            print("KEY INSIGHTS")
            print(f"{'-'*80}")
            
            if not self.summary_stats.empty:
                # Best performing setup by regime
                best_by_regime = self.summary_stats.loc[self.summary_stats.groupby('volatility_regime')['total_net_r'].idxmax()]
                
                for _, row in best_by_regime.iterrows():
                    print(f"{row['volatility_regime']} Volatility: Best setup is '{row['setup']}' "
                          f"({row['win_rate']:.1f}% win rate, {row['total_net_r']:.1f}R total)")
                
                # Regime recommendations
                high_vol_data = self.summary_stats[self.summary_stats['volatility_regime'] == 'HIGH']
                low_vol_data = self.summary_stats[self.summary_stats['volatility_regime'] == 'LOW']
                
                if not high_vol_data.empty:
                    high_vol_avg_r = high_vol_data['total_net_r'].mean()
                    print(f"\nHigh Volatility Performance: {high_vol_avg_r:.1f}R average")
                
                if not low_vol_data.empty:
                    low_vol_avg_r = low_vol_data['total_net_r'].mean()
                    print(f"Low Volatility Performance: {low_vol_avg_r:.1f}R average")
            
            print("="*80)
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
    
    def save_summary(self, filepath: str = "logs/garch_overlay_summary.csv") -> bool:
        """
        Save summary statistics to CSV file.
        
        Args:
            filepath: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.summary_stats.empty:
                print("❌ No summary data to save")
                return False
            
            # Ensure logs directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save summary table
            self.summary_stats.to_csv(filepath, index=False)
            print(f"✓ Summary saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving summary: {e}")
            return False
    
    def _create_sample_trade_data(self, filepath: str) -> None:
        """
        Create a sample trade journal file for demonstration.
        
        Args:
            filepath: Path where to create the sample file
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Sample trade data
            sample_data = [
                ['2024-01-15', 'SPY', 'Breakout_T1', 'LONG', 450.50, 455.20, 'win', 2.1, 'Clean breakout above resistance'],
                ['2024-01-16', 'SPY', 'Pullback_T2', 'LONG', 452.30, 450.80, 'loss', -0.8, 'Failed to hold support'],
                ['2024-01-18', 'GLD', 'Reversal_T1', 'SHORT', 198.50, 195.20, 'win', 1.5, 'Strong reversal at key level'],
                ['2024-01-22', 'SPY', 'Breakout_T1', 'LONG', 448.20, 453.90, 'win', 2.3, 'Volume confirmation'],
                ['2024-01-25', 'GLD', 'Pullback_T2', 'LONG', 196.80, 199.40, 'win', 1.2, 'Tier 2 setup worked'],
                ['2024-01-30', 'SPY', 'Reversal_T1', 'SHORT', 451.60, 449.20, 'win', 1.8, 'Perfect rejection'],
            ]
            
            # Create DataFrame and save
            columns = ['date', 'symbol', 'setup', 'direction', 'entry_price', 'exit_price', 'result', 'r_multiple', 'notes']
            sample_df = pd.DataFrame(sample_data, columns=columns)
            sample_df.to_csv(filepath, index=False)
            
            print(f"✓ Created sample trade journal: {filepath}")
            print("You can edit this file to add your actual trade data")
            
        except Exception as e:
            print(f"❌ Error creating sample data: {e}")


def run_overlay_analysis(
    trade_file: str = "logs/trader_journal.csv",
    garch_file: str = "logs/garch_forecast_log.csv",
    output_file: str = "logs/garch_overlay_summary.csv",
    save_results: bool = True
) -> Dict[str, Any]:
    """
    Run the complete GARCH overlay analysis.
    
    Args:
        trade_file: Path to trade journal CSV
        garch_file: Path to GARCH forecast CSV
        output_file: Path for summary output CSV
        save_results: Whether to save results to file
        
    Returns:
        Analysis results dictionary
    """
    print("Starting GARCH Volatility Overlay Analysis...")
    
    # Initialize analyzer
    analyzer = GARCHOverlayAnalyzer()
    
    # Load data
    if not analyzer.load_trade_data(trade_file):
        return {}
    
    if not analyzer.load_garch_data(garch_file):
        return {}
    
    # Merge datasets
    if not analyzer.merge_datasets():
        return {}
    
    # Analyze performance
    results = analyzer.analyze_performance()
    
    if results:
        # Print report
        analyzer.print_summary_report(results)
        
        # Save results if requested
        if save_results:
            analyzer.save_summary(output_file)
    
    return results


if __name__ == "__main__":
    """Run overlay analysis as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze trading performance with GARCH volatility overlay")
    parser.add_argument('--trades', default='logs/trader_journal.csv', help='Path to trade journal CSV')
    parser.add_argument('--garch', default='logs/garch_forecast_log.csv', help='Path to GARCH forecast CSV')
    parser.add_argument('--output', default='logs/garch_overlay_summary.csv', help='Output summary CSV path')
    parser.add_argument('--no-save', action='store_true', help='Do not save results to file')
    
    args = parser.parse_args()
    
    # Run analysis
    results = run_overlay_analysis(
        trade_file=args.trades,
        garch_file=args.garch,
        output_file=args.output,
        save_results=not args.no_save
    ) 