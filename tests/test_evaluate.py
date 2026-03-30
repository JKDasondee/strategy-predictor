import numpy as np
import pytest
from src.evaluate import rmse, mape, mae, r_squared, directional_accuracy
from src.baseline import BaselineResult

def test_rmse_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0

def test_rmse_known():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.1, 2.1, 3.1])
    assert abs(rmse(yt, yp) - 0.1) < 1e-6

def test_mape():
    yt = np.array([100.0, 200.0])
    yp = np.array([110.0, 180.0])
    assert abs(mape(yt, yp) - 10.0) < 1e-6

def test_mae():
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.5, 2.5, 3.5])
    assert abs(mae(yt, yp) - 0.5) < 1e-6

def test_r_squared_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(r_squared(y, y) - 1.0) < 1e-6

def test_directional_accuracy():
    yt = np.array([1.0, -1.0, 1.0, -1.0])
    yp = np.array([0.5, -0.5, -0.5, 0.5])
    assert abs(directional_accuracy(yt, yp) - 50.0) < 1e-6
