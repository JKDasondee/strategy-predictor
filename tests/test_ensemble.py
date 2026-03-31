import numpy as np
import pytest
from src.baseline import BaselineResult
from src.ensemble import WeightedEnsemble, StackedEnsemble

def _results() -> list[BaselineResult]:
    np.random.seed(42)
    y = np.random.randn(50)
    return [
        BaselineResult("model_a", y, y + np.random.randn(50) * 0.1),
        BaselineResult("model_b", y, y + np.random.randn(50) * 0.3),
        BaselineResult("model_c", y, y + np.random.randn(50) * 0.5),
    ]

def test_weighted_ensemble_fit():
    results = _results()
    ens = WeightedEnsemble()
    ens.fit(results)
    assert len(ens.weights) == 3
    total = sum(w.weight for w in ens.weights)
    assert abs(total - 1.0) < 1e-4

def test_weighted_ensemble_predict():
    results = _results()
    ens = WeightedEnsemble()
    ens.fit(results)
    r = ens.predict(results)
    assert r.name == "ensemble"
    assert len(r.y_pred) == 50

def test_weighted_favors_better_model():
    results = _results()
    ens = WeightedEnsemble()
    ens.fit(results)
    w_map = {w.name: w.weight for w in ens.weights}
    assert w_map["model_a"] > w_map["model_c"]

def test_stacked_ensemble():
    results = _results()
    se = StackedEnsemble()
    se.fit(results)
    r = se.predict(results)
    assert r.name == "stacked"
    assert len(r.y_pred) == 50
    assert se.coefs is not None
    assert abs(se.coefs.sum() - 1.0) < 1e-4

def test_single_model_ensemble():
    np.random.seed(42)
    y = np.random.randn(30)
    results = [BaselineResult("only", y, y + 0.01)]
    ens = WeightedEnsemble()
    ens.fit(results)
    assert len(ens.weights) == 1
    assert abs(ens.weights[0].weight - 1.0) < 1e-6
