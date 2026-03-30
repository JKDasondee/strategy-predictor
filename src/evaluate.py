import numpy as np
from typing import Optional
from src.baseline import BaselineResult

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred) & (np.abs(y_true) > eps)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))

def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    return float(np.mean(np.sign(yt) == np.sign(yp)) * 100)

def comparison_table(results: list[BaselineResult]) -> str:
    header = f"{'Model':<20} {'RMSE':>10} {'MAE':>10} {'MAPE%':>10} {'R²':>10} {'Dir%':>10}"
    sep = "-" * len(header)
    lines = [header, sep]
    for r in results:
        lines.append(
            f"{r.name:<20} "
            f"{rmse(r.y_true, r.y_pred):>10.6f} "
            f"{mae(r.y_true, r.y_pred):>10.6f} "
            f"{mape(r.y_true, r.y_pred):>10.2f} "
            f"{r_squared(r.y_true, r.y_pred):>10.4f} "
            f"{directional_accuracy(r.y_true, r.y_pred):>10.1f}"
        )
    return "\n".join(lines)

def beats_baseline(model_result: BaselineResult, baseline_result: BaselineResult) -> dict:
    mr = rmse(model_result.y_true, model_result.y_pred)
    br = rmse(baseline_result.y_true, baseline_result.y_pred)
    return {
        "model_rmse": mr,
        "baseline_rmse": br,
        "beats_baseline": mr < br,
        "improvement_pct": (1 - mr / br) * 100 if br > 0 else 0.0,
    }
