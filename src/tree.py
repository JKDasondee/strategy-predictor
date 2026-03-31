import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from typing import Optional
from dataclasses import dataclass, field

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from src.baseline import BaselineResult

@dataclass
class TreeConfig:
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 20
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

class XGBoostModel:
    def __init__(self, cfg: Optional[TreeConfig] = None) -> None:
        self.name = "xgboost"
        self.cfg = cfg or TreeConfig()
        self.model = None
        self.scaler = StandardScaler()

    def predict(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        asset: str,
    ) -> BaselineResult:
        if not HAS_XGB:
            raise ImportError("xgboost not installed: pip install xgboost")

        Xtr = self._prep(X_train, fit=True)
        Xte = self._prep(X_test, fit=False)
        ytr = y_train[asset].values
        yte = y_test[asset].values

        mask = ~np.isnan(ytr)
        dtrain = xgb.DMatrix(Xtr[mask], label=ytr[mask])
        dval = xgb.DMatrix(Xte, label=yte)

        params = {
            "objective": "reg:squarederror",
            "max_depth": self.cfg.max_depth,
            "learning_rate": self.cfg.learning_rate,
            "subsample": self.cfg.subsample,
            "colsample_bytree": self.cfg.colsample_bytree,
            "reg_alpha": self.cfg.reg_alpha,
            "reg_lambda": self.cfg.reg_lambda,
            "verbosity": 0,
        }

        self.model = xgb.train(
            params, dtrain,
            num_boost_round=self.cfg.n_estimators,
            evals=[(dval, "val")],
            early_stopping_rounds=self.cfg.early_stopping_rounds,
            verbose_eval=False,
        )

        pred = self.model.predict(dval)
        return BaselineResult(self.name, yte, pred)

    def feature_importance(self, top_n: int = 20) -> dict[str, float]:
        if self.model is None:
            return {}
        imp = self.model.get_score(importance_type="gain")
        sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_imp[:top_n])

    def _prep(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        arr = X.values.astype(np.float64)
        arr = np.nan_to_num(arr, nan=0.0)
        if fit:
            return self.scaler.fit_transform(arr)
        return self.scaler.transform(arr)

class LightGBMModel:
    def __init__(self, cfg: Optional[TreeConfig] = None) -> None:
        self.name = "lightgbm"
        self.cfg = cfg or TreeConfig()
        self.model = None
        self.scaler = StandardScaler()

    def predict(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame,
        asset: str,
    ) -> BaselineResult:
        if not HAS_LGB:
            raise ImportError("lightgbm not installed: pip install lightgbm")

        Xtr = self._prep(X_train, fit=True)
        Xte = self._prep(X_test, fit=False)
        ytr = y_train[asset].values
        yte = y_test[asset].values

        mask = ~np.isnan(ytr)

        params = {
            "objective": "regression",
            "metric": "mse",
            "max_depth": self.cfg.max_depth,
            "learning_rate": self.cfg.learning_rate,
            "subsample": self.cfg.subsample,
            "colsample_bytree": self.cfg.colsample_bytree,
            "reg_alpha": self.cfg.reg_alpha,
            "reg_lambda": self.cfg.reg_lambda,
            "verbose": -1,
            "n_jobs": -1,
        }

        dtrain = lgb.Dataset(Xtr[mask], label=ytr[mask])
        dval = lgb.Dataset(Xte, label=yte, reference=dtrain)

        self.model = lgb.train(
            params, dtrain,
            num_boost_round=self.cfg.n_estimators,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)],
        )

        pred = self.model.predict(Xte)
        return BaselineResult(self.name, yte, pred)

    def feature_importance(self, top_n: int = 20) -> list[tuple[str, float]]:
        if self.model is None:
            return []
        imp = self.model.feature_importance(importance_type="gain")
        names = self.model.feature_name()
        paired = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)
        return paired[:top_n]

    def _prep(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        arr = X.values.astype(np.float64)
        arr = np.nan_to_num(arr, nan=0.0)
        if fit:
            return self.scaler.fit_transform(arr)
        return self.scaler.transform(arr)

def cross_validate_tree(
    model_cls: type,
    X: pd.DataFrame,
    y: pd.DataFrame,
    asset: str,
    n_splits: int = 3,
    cfg: Optional[TreeConfig] = None,
) -> list[BaselineResult]:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    for train_idx, test_idx in tscv.split(X):
        m = model_cls(cfg)
        r = m.predict(
            X.iloc[train_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[test_idx],
            asset,
        )
        results.append(r)
    return results
