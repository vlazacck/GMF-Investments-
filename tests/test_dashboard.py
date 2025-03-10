import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from dashboard.dash import load_data, calculate_cvar, objective_function, calculate_maximum_drawdown
from scipy.optimize import minimize
import sys
import os

# Add the project root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import functions from dashboard.dash
from dashboard.dash import load_data, calculate_cvar, objective_function, calculate_maximum_drawdown


# Mock Data for Testing
mock_returns = pd.DataFrame({
    "TSLA": [0.01, -0.02, 0.015, 0.03, -0.01],
    "BND": [-0.005, 0.002, 0.001, -0.003, 0.004],
    "SPY": [0.02, -0.01, 0.015, 0.025, -0.005]
})

mock_weights = np.array([0.5, 0.3, 0.2])  # Example portfolio allocation

# ✅ Unit Tests
def test_load_data():
    data = load_data()
    assert not data.empty, "Market data should not be empty"
    assert isinstance(data, pd.DataFrame), "Data should be a pandas DataFrame"

def test_calculate_cvar():
    cvar = calculate_cvar(mock_returns, mock_weights)
    assert cvar < 0, "CVaR should be a negative value (representing risk)"

def test_objective_function():
    lambda_cvar = 10
    obj_value = objective_function(mock_weights, mock_returns, lambda_cvar)
    assert isinstance(obj_value, float), "Objective function should return a float"

def test_calculate_maximum_drawdown():
    max_dd = calculate_maximum_drawdown(mock_returns, mock_weights)
    assert max_dd < 0, "Maximum drawdown should be negative (indicating loss)"

# ✅ Integration Tests
def test_portfolio_optimization():
    num_assets = mock_returns.shape[1]
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(num_assets)]
    initial_weights = np.array([1/num_assets] * num_assets)
    lambda_cvar = 10

    optimized = minimize(objective_function, initial_weights, args=(mock_returns, lambda_cvar), 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    assert optimized.success, "Portfolio optimization should succeed"
    assert np.isclose(np.sum(optimized.x), 1), "Optimized weights should sum to 1"

if __name__ == "__main__":
    pytest.main()
