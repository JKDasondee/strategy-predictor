import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from src.model import SequenceDataset, TrainConfig
from typing import Optional

class NBEATSBlock(nn.Module):
    def __init__(self, in_dim: int, theta_dim: int, hidden: int = 128, layers: int = 4) -> None:
        super().__init__()
        stack = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            stack += [nn.Linear(hidden, hidden), nn.ReLU()]
        self.fc = nn.Sequential(*stack)
        self.theta_b = nn.Linear(hidden, theta_dim)
        self.theta_f = nn.Linear(hidden, theta_dim)
        self.backcast = nn.Linear(theta_dim, in_dim)
        self.forecast = nn.Linear(theta_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.fc(x)
        tb = self.theta_b(h)
        tf = self.theta_f(h)
        return self.backcast(tb), self.forecast(tf).squeeze(-1)

class NBEATSStack(nn.Module):
    def __init__(self, n_blocks: int, in_dim: int, theta_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(in_dim, theta_dim, hidden) for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        forecast = torch.zeros(x.shape[0], device=x.device)
        for block in self.blocks:
            b, f = block(residual)
            residual = residual - b
            forecast = forecast + f
        return residual, forecast

class NBEATS(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        n_stacks: int = 2,
        n_blocks: int = 3,
        theta_dim: int = 16,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        flat_dim = input_dim * seq_len
        self.flat_dim = flat_dim
        self.stacks = nn.ModuleList([
            NBEATSStack(n_blocks, flat_dim, theta_dim, hidden)
            for _ in range(n_stacks)
        ])
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        b = x.shape[0]
        x_flat = x.reshape(b, -1)
        residual = x_flat
        forecast = torch.zeros(b, device=x.device)
        for stack in self.stacks:
            residual = self.drop(residual)
            r, f = stack(residual)
            residual = r
            forecast = forecast + f
        return forecast

def train_nbeats(
    model: NBEATS,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Optional[TrainConfig] = None,
) -> dict:
    cfg = cfg or TrainConfig(epochs=100, lr=1e-3, patience=15)
    ds = SequenceDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        tl = float(np.mean(losses))
        history["train_loss"].append(tl)
        history["val_loss"].append(val_loss)
        sched.step(val_loss)

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    return {
        "best_val_loss": best_val,
        "epochs_trained": len(history["train_loss"]),
        "history": history,
    }

def mc_dropout_nbeats(
    model: NBEATS,
    X: np.ndarray,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.train()
    X_t = torch.FloatTensor(X)
    preds = []
    with torch.no_grad():
        for _ in range(n_samples):
            p = model(X_t).numpy()
            preds.append(p)
    preds = np.stack(preds)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)
    return mean, mean - 2 * std, mean + 2 * std
