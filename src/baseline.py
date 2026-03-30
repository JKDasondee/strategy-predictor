import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Optional
from dataclasses import dataclass

@dataclass
class BaselineResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray

class NaiveBaseline:
    def __init__(self) -> None:
        self.name = "naive"

    def predict(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.DataFrame, y_test: pd.DataFrame,
                asset: str) -> BaselineResult:
        # predict next 7d return = last observed 7d return
        r7_cols = [c for c in X_test.columns if c[0] == "r7"]
        if asset in [c[1] for c in r7_cols]:
            pred = X_test[("r7", asset)].values
        else:
            pred = np.zeros(len(X_test))
        return BaselineResult(self.name, y_test[asset].values, pred)

class MeanReversion:
    def __init__(self, window: int = 14) -> None:
        self.name = "mean_reversion"
        self.w = window

    def predict(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.DataFrame, y_test: pd.DataFrame,
                asset: str) -> BaselineResult:
        r7_cols = [c for c in X_test.columns if c[0] == "r7"]
        if asset in [c[1] for c in r7_cols]:
            r = X_test[("r7", asset)].values
            mu = np.mean(X_train[("r7", asset)].values[-self.w:])
            pred = mu - 0.5 * (r - mu)
        else:
            pred = np.zeros(len(X_test))
        return BaselineResult(self.name, y_test[asset].values, pred)

class LinearBaseline:
    def __init__(self, alpha: float = 1.0) -> None:
        self.name = "ridge"
        self.alpha = alpha
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)

    def predict(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                y_train: pd.DataFrame, y_test: pd.DataFrame,
                asset: str) -> BaselineResult:
        Xtr = self._flat(X_train)
        Xte = self._flat(X_test)
        ytr = y_train[asset].values
        yte = y_test[asset].values

        Xtr_s = self.scaler.fit_transform(Xtr)
        Xte_s = self.scaler.transform(Xte)

        mask = ~np.isnan(ytr) & ~np.any(np.isnan(Xtr_s), axis=1)
        self.model.fit(Xtr_s[mask], ytr[mask])
        pred = self.model.predict(Xte_s)
        return BaselineResult(self.name, yte, pred)

    def _flat(self, X: pd.DataFrame) -> np.ndarray:
        arr = X.values.astype(np.float64)
        arr = np.nan_to_num(arr, nan=0.0)
        return arr
