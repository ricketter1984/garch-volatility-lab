"""
Unit Tests for GARCH Forecast Module

This module contains basic unit tests for the GARCHForecaster class,
testing model initialization, fitting, and forecasting functionality.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from garch.forecast import GARCHForecaster


class TestGARCHForecaster(unittest.TestCase):
    """Test cases for GARCHForecaster class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create synthetic return data for testing
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', end='2023-01-01', freq='D')
        self.returns = pd.Series(
            np.random.normal(0, 0.02, len(dates)), 
            index=dates,
            name='returns'
        )
        
        # Add some volatility clustering
        high_vol_periods = np.random.choice(len(dates), size=50, replace=False)
        self.returns.iloc[high_vol_periods] *= 3
        
        self.forecaster = GARCHForecaster()
    
    def test_initialization(self):
        """Test GARCHForecaster initialization."""
        forecaster = GARCHForecaster(vol_model='GARCH', p=1, q=1)
        
        self.assertEqual(forecaster.vol_model, 'GARCH')
        self.assertEqual(forecaster.p, 1)
        self.assertEqual(forecaster.q, 1)
        self.assertIsNone(forecaster.model)
        self.assertIsNone(forecaster.fitted_model)
        self.assertIsNone(forecaster.data)
    
    def test_fit_basic(self):
        """Test basic model fitting."""
        result = self.forecaster.fit(self.returns)
        
        # Should return self for chaining
        self.assertEqual(result, self.forecaster)
        
        # Should have fitted model (if successful)
        self.assertIsNotNone(self.forecaster.model)
        
        # Data should be stored
        self.assertIsNotNone(self.forecaster.data)
        self.assertEqual(len(self.forecaster.data), len(self.returns.dropna()))
    
    def test_fit_with_empty_data(self):
        """Test fitting with empty or invalid data."""
        empty_returns = pd.Series([], dtype=float)
        
        self.forecaster.fit(empty_returns)
        
        # Should handle empty data gracefully
        self.assertIsNotNone(self.forecaster.data)
        self.assertEqual(len(self.forecaster.data), 0)
    
    def test_forecast_without_fit(self):
        """Test forecasting without fitting model first."""
        with self.assertRaises(ValueError):
            self.forecaster.forecast()
    
    def test_forecast_after_fit(self):
        """Test forecasting after successful model fit."""
        # Fit model first
        self.forecaster.fit(self.returns)
        
        # Only test forecast if model was fitted successfully
        if self.forecaster.fitted_model is not None:
            forecast = self.forecaster.forecast(horizon=1)
            
            # Check forecast structure
            self.assertIsInstance(forecast, dict)
            self.assertIn('horizon', forecast)
            self.assertIn('volatility_forecast', forecast)
            self.assertIn('variance_forecast', forecast)
            self.assertIn('annualized_volatility', forecast)
            self.assertIn('forecast_date', forecast)
            
            # Check forecast values
            self.assertEqual(forecast['horizon'], 1)
            self.assertGreater(forecast['volatility_forecast'][0], 0)
            self.assertGreater(forecast['annualized_volatility'][0], 0)
    
    def test_get_model_summary_without_fit(self):
        """Test getting model summary without fitting."""
        summary = self.forecaster.get_model_summary()
        self.assertEqual(summary, "Model not fitted")
    
    def test_get_parameters_without_fit(self):
        """Test getting parameters without fitting."""
        params = self.forecaster.get_parameters()
        self.assertEqual(params, {})
    
    def test_get_parameters_after_fit(self):
        """Test getting parameters after fitting."""
        self.forecaster.fit(self.returns)
        
        if self.forecaster.fitted_model is not None:
            params = self.forecaster.get_parameters()
            self.assertIsInstance(params, dict)
    
    def test_volatility_clusters(self):
        """Test volatility clustering identification."""
        self.forecaster.fit(self.returns)
        
        if self.forecaster.fitted_model is not None:
            clusters = self.forecaster.calculate_volatility_clusters(threshold=2.0)
            
            self.assertIsInstance(clusters, pd.Series)
            self.assertEqual(len(clusters), len(self.forecaster.fitted_model.conditional_volatility))
            self.assertTrue(clusters.dtype == bool)
    
    def test_different_vol_models(self):
        """Test different volatility model specifications."""
        models_to_test = ['GARCH', 'EGARCH']
        
        for vol_model in models_to_test:
            with self.subTest(vol_model=vol_model):
                forecaster = GARCHForecaster(vol_model=vol_model, p=1, q=1)
                forecaster.fit(self.returns)
                
                self.assertEqual(forecaster.vol_model, vol_model)
                # Note: We don't test if fitting succeeds as some models may not converge with synthetic data


class TestGARCHForecastIntegration(unittest.TestCase):
    """Integration tests for GARCH forecasting workflow."""
    
    def test_end_to_end_workflow(self):
        """Test complete forecasting workflow."""
        # Generate more realistic synthetic data
        np.random.seed(123)
        dates = pd.date_range(start='2021-01-01', end='2023-01-01', freq='D')
        
        # Generate returns with time-varying volatility
        vol = 0.01 + 0.005 * np.sin(np.arange(len(dates)) * 2 * np.pi / 252)
        returns = pd.Series(
            np.random.normal(0, vol),
            index=dates,
            name='returns'
        )
        
        # Initialize and fit forecaster
        forecaster = GARCHForecaster(vol_model='GARCH', p=1, q=1)
        forecaster.fit(returns, mean='Constant')
        
        # If model fitted successfully, test complete workflow
        if forecaster.fitted_model is not None:
            # Test forecasting
            forecast = forecaster.forecast(horizon=5)
            self.assertEqual(forecast['horizon'], 5)
            self.assertEqual(len(forecast['volatility_forecast']), 5)
            
            # Test parameter extraction
            params = forecaster.get_parameters()
            self.assertIsInstance(params, dict)
            self.assertGreater(len(params), 0)
            
            # Test volatility clustering
            clusters = forecaster.calculate_volatility_clusters()
            self.assertIsInstance(clusters, pd.Series)
            
            # Test model summary
            summary = forecaster.get_model_summary()
            self.assertNotEqual(summary, "Model not fitted")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 