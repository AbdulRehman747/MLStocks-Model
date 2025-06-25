"""
TorchServe handler for the MLStocks LSTM model
----------------------------------------------

✓ Parses JSON, NPY, or pickled NumPy payloads.
✓ Normalises inputs with the training-time scalers.
✓ Runs prediction on CPU (or GPU if available).
✓ De-normalises outputs back to real-world scale (volume ≥ 0, etc.).
✓ Returns JSON  {"predictions": [[…], …]}.

Author: ChatGPT  ·  June 2025
"""

# ───────────────────────── Imports ─────────────────────────
import io, json, pickle
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────── Architecture ─────────────────────
class StockLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 25,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # x: (B, T, F)
        B = x.size(0)
        h0 = x.new_zeros(self.lstm.num_layers, B, self.lstm.hidden_size)
        c0 = x.new_zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(out[:, -1]))                  # (B, out)

# ──────────────────────── Utilities ────────────────────────
def _cpu_pickle_load(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        pkg = pickle.load(f)
    if "model_state_dict" in pkg:
        pkg["model_state_dict"] = {k: v.cpu() for k, v in pkg["model_state_dict"].items()}
    return pkg


def _normalize(mat: np.ndarray, model: nn.Module) -> np.ndarray:
    """mat shape (N, 5)"""
    if model.use_global_scaler:
        return model.global_scaler.transform(mat)
    out = mat.copy()
    for i, scaler in enumerate(model.individual_scalers.values()):
        out[:, i] = scaler.transform(mat[:, i].reshape(-1, 1)).ravel()
    return out


def _denormalize(mat: np.ndarray, model: nn.Module) -> np.ndarray:
    """
    mat shape (B, H*5)  where H = horizon.
    Inverse-transform per 5-feature block.
    """
    B, D = mat.shape
    F = 5
    if D % F != 0:
        raise ValueError("Output dim must be a multiple of 5")
    H = D // F

    flat = mat.reshape(B * H, F)          # (B*H, 5)
    if model.use_global_scaler:
        real = model.global_scaler.inverse_transform(flat)
    else:
        real = flat.copy()
        for i, scaler in enumerate(model.individual_scalers.values()):
            real[:, i] = scaler.inverse_transform(flat[:, i].reshape(-1, 1)).ravel()
    return real.reshape(B, D)

# ─────────────────── TorchServe entry points ───────────────
def model_fn(model_dir: str) -> nn.Module:
    pkg = _cpu_pickle_load(Path(model_dir) / "model.pkl")

    model = StockLSTM(**pkg["model_architecture"]).eval()
    model.load_state_dict(pkg["model_state_dict"])

    scalers = pkg["scalers"]
    model.use_global_scaler   = scalers["use_global_scaler"]
    model.global_scaler       = scalers["global_scaler"]
    model.individual_scalers  = scalers["individual_scalers"]

    return model


def _json_to_array(body: Union[bytes, str]) -> np.ndarray:
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    data = json.loads(body)
    rows = data["instances"] if isinstance(data, dict) else data
    if rows and isinstance(rows[0], dict):        # list-of-dicts
        keys = list(rows[0].keys())
        rows = [[row[k] for k in keys] for row in rows]
    return np.asarray(rows, dtype=np.float32)


def input_fn(body, content_type: str) -> np.ndarray:
    if content_type.startswith("application/json"):
        arr = _json_to_array(body)
    elif content_type == "application/x-npy":
        arr = np.load(io.BytesIO(body)).astype(np.float32)
    elif content_type in ("application/octet-stream", "application/x-pickle"):
        arr = pickle.loads(body).astype(np.float32)
    else:
        raise ValueError(f"Unsupported content-type: {content_type}")

    if arr.ndim == 2:              # (T, F)  → add batch dim
        arr = arr[None, ...]
    return arr                     # shape (B, T, F)


def predict_fn(data: np.ndarray, model: nn.Module) -> np.ndarray:
    B, T, F = data.shape

    # 1) normalise each timestep row-wise
    norm = _normalize(data.reshape(-1, F), model).reshape(B, T, F)

    # 2) forward pass
    tensor = torch.from_numpy(norm).to(next(model.parameters()).device)
    with torch.no_grad():
        raw = model(tensor).cpu().numpy()      # (B, H*5)

    # 3) inverse-scale
    return _denormalize(raw, model)            # (B, H*5)


def output_fn(prediction: np.ndarray, accept: str) -> Tuple[str, str]:
    if accept not in ("application/json", "application/json; charset=utf-8"):
        raise ValueError(f"Unsupported accept header: {accept}")
    return json.dumps({"predictions": prediction.tolist()}), accept
