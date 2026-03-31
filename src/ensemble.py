import numpy as np
from scipy.optimize import minimize
from typing import Optional
from dataclasses import dataclass
from src.baseline import BaselineResult
from src.evaluate import rmse

@dataclass
class EnsembleWeight:
    name: str
    weight: float

class WeightedEnsemble:
    def __init__(self) -> None:
        self.name = "ensemble"
        self.weights: list[EnsembleWeight] = []

    def fit(self, results: list[BaselineResult]) -> None:
        n = len(results)
        if n == 0:
            return
        if n == 1:
            self.weights = [EnsembleWeight(results[0].name, 1.0)]
            return

        preds = np.stack([r.y_pred for r in results])
        y_true = results[0].y_true

        def obj(w: np.ndarray) -> float:
            w = np.abs(w) / np.abs(w).sum()
            blend = (preds.T @ w)
            return rmse(y_true, blend)

        w0 = np.ones(n) / n
        res = minimize(obj, w0, method="Nelder-Mead",
                       options={"maxiter": 1000, "xatol": 1e-8})
        w_opt = np.abs(res.x) / np.abs(res.x).sum()

        self.weights = [
            EnsembleWeight(r.name, float(w))
            for r, w in zip(results, w_opt)
        ]

    def predict(self, results: list[BaselineResult]) -> BaselineResult:
        if not self.weights:
            self.fit(results)

        w_map = {ew.name: ew.weight for ew in self.weights}
        preds = []
        ws = []
        for r in results:
            w = w_map.get(r.name, 0.0)
            preds.append(r.y_pred * w)
            ws.append(w)

        blend = np.sum(preds, axis=0)
        return BaselineResult(self.name, results[0].y_true, blend)

    def summary(self) -> str:
        lines = ["Ensemble Weights:"]
        for ew in sorted(self.weights, key=lambda x: x.weight, reverse=True):
            lines.append(f"  {ew.name:<20} {ew.weight:.4f}")
        return "\n".join(lines)

class StackedEnsemble:
    def __init__(self) -> None:
        self.name = "stacked"
        self.coefs: Optional[np.ndarray] = None

    def fit(self, results: list[BaselineResult]) -> None:
        preds = np.stack([r.y_pred for r in results]).T  # (n_samples, n_models)
        y = results[0].y_true

        mask = ~np.isnan(y)
        X = preds[mask]
        y_clean = y[mask]

        # closed-form ridge: (X'X + λI)^-1 X'y
        lam = 0.01
        XtX = X.T @ X + lam * np.eye(X.shape[1])
        Xty = X.T @ y_clean
        self.coefs = np.linalg.solve(XtX, Xty)
        self.coefs = np.maximum(self.coefs, 0)
        s = self.coefs.sum()
        if s > 0:
            self.coefs /= s

    def predict(self, results: list[BaselineResult]) -> BaselineResult:
        if self.coefs is None:
            self.fit(results)
        preds = np.stack([r.y_pred for r in results]).T
        blend = preds @ self.coefs
        return BaselineResult(self.name, results[0].y_true, blend)

    def summary(self) -> str:
        if self.coefs is None:
            return "Stacked: not fitted"
        return f"Stacked coefficients: {self.coefs.round(4).tolist()}"
