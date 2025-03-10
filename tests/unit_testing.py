import pytest
import numpy as np
import pandas as pd
from dashboard.dash import (
    load_data, calculate_cvar, objective_function, calculate_maximum_drawdown
)

# Sample data for testing
returns_sample = pd.DataFrame({
    "TSLA": [-0.2, -0.15, -0.1, -0.08, -0.06],
    "BND": [-0.05, -0.04, -0.03, -0.02, -0.01],
    "SPY": [-0.15, -0.1, -0.08, -0.06, -0.04]
})





weights_sample = np.array([0.4, 0.3, 0.3])

def test_load_data():
    data = load_data()
    assert isinstance(data, pd.DataFrame)
    assert not data.empty  # Ensure data is loaded

def test_calculate_cvar():
    cvar = calculate_cvar(returns_sample, weights_sample)
    assert isinstance(cvar, float)
    assert cvar < 0  # CVaR should be negative (risk measure)

def test_calculate_maximum_drawdown():
    mdd = calculate_maximum_drawdown(returns_sample, weights_sample)
    assert isinstance(mdd, float)
    assert -1 <= mdd <= 0  # MDD is always between -1 and 0

def test_objective_function():
    lambda_cvar = 10
    obj_value = objective_function(weights_sample, returns_sample, lambda_cvar)
    assert isinstance(obj_value, float)
