import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Optional
from dataclasses import dataclass

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[i], self.y[i]

class StrategyLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out).squeeze(-1)

@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32
    weight_decay: float = 1e-5
    patience: int = 15
    min_delta: float = 1e-6

def train(
    model: StrategyLSTM,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Optional[TrainConfig] = None,
) -> dict:
    cfg = cfg or TrainConfig()
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

        train_loss = np.mean(losses)
        history["train_loss"].append(train_loss)
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

def mc_dropout_predict(
    model: StrategyLSTM,
    X: np.ndarray,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.train()  # keep dropout active
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

def predict_deterministic(
    model: StrategyLSTM,
    X: np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model(torch.FloatTensor(X)).numpy()
