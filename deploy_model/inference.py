"""
TorchServe handler for the MLStocks LSTM model
----------------------------------------------

✓  Parses JSON, NPY, or pickled NumPy payloads.
✓  Normalizes features with the exact scalers used at training time.
✓  Runs prediction on CPU (or GPU if available).
✓  De-normalizes the network outputs to real-world scale.
✓  Returns JSON:  {"predictions": [[…], …]}

Author: ChatGPT (June 2025)
"""

import io
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from typing import Union


# ────────────────────────────────────────────────────────────────────────────────
# 1.  Model architecture
# ────────────────────────────────────────────────────────────────────────────────
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
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        bsz = x.size(0)
        device = x.device
        h0 = torch.zeros(self.lstm.num_layers, bsz, self.lstm.hidden_size, device=device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(out[:, -1]))  # (batch, output_size)


# ────────────────────────────────────────────────────────────────────────────────
# 2.  Utility helpers
# ────────────────────────────────────────────────────────────────────────────────
def _cpu_pickle_load(path: Path) -> Dict[str, Any]:
    """Always load a pickle file on CPU, even if saved on GPU."""
    with path.open("rb") as f:
        pkg = pickle.load(f)
    if "model_state_dict" in pkg:
        pkg["model_state_dict"] = {k: v.cpu() for k, v in pkg["model_state_dict"].items()}
    return pkg


def _normalize(arr: np.ndarray, model: nn.Module) -> np.ndarray:
    """Normalize with either a global scaler or per-feature scalers."""
    if model.use_global_scaler:
        return model.global_scaler.transform(arr)

    out = arr.copy()
    for i, scaler in enumerate(model.individual_scalers.values()):
        out[:, i] = scaler.transform(arr[:, i].reshape(-1, 1)).ravel()
    return out


def _denormalize(arr: np.ndarray, model: nn.Module) -> np.ndarray:
    """Inverse-transform predictions back to real scale."""
    if model.use_global_scaler:
        return model.global_scaler.inverse_transform(arr)

    out = arr.copy()
    for i, scaler in enumerate(model.individual_scalers.values()):
        out[:, i] = scaler.inverse_transform(arr[:, i].reshape(-1, 1)).ravel()
    return out


# ────────────────────────────────────────────────────────────────────────────────
# 3.  TorchServe entry points
# ────────────────────────────────────────────────────────────────────────────────
def model_fn(model_dir: str) -> nn.Module:
    """
    Called once at model load time. Returns the model in eval mode
    with scaler objects hanging off it for downstream use.
    """
    pkg = _cpu_pickle_load(Path(model_dir) / "model.pkl")

    # Rebuild network and load weights
    model_cfg = pkg["model_architecture"]
    model = StockLSTM(**model_cfg).eval()
    model.load_state_dict(pkg["model_state_dict"])

    # Attach scalers so input_fn / predict_fn can access them
    scalers = pkg["scalers"]  # must contain the three keys below
    model.use_global_scaler = scalers["use_global_scaler"]
    model.global_scaler = scalers["global_scaler"]
    model.individual_scalers = scalers["individual_scalers"]

    return model


def _json_to_array(body: Union[bytes, str]) -> np.ndarray:
    """Parse the JSON body and return an ndarray (n_rows, n_features)."""
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")

    data = json.loads(body)

    # Accept either {"instances": [...]} or bare list/array
    rows = data["instances"] if isinstance(data, dict) else data

    # Handle list-of-dicts ↦ list-of-lists (stable key order)
    if rows and isinstance(rows[0], dict):
        keys = list(rows[0].keys())  # assume consistent order
        rows = [[row[k] for k in keys] for row in rows]

    return np.asarray(rows, dtype=np.float32)


def input_fn(body, content_type: str) -> np.ndarray:
    """
    Parse the HTTP request body and return a *numpy* array.
    TorchServe will pass this object untouched to predict_fn.
    """
    if content_type.startswith("application/json"):
        arr = _json_to_array(body)

    elif content_type == "application/x-npy":
        arr = np.load(io.BytesIO(body)).astype(np.float32)

    elif content_type in ("application/octet-stream", "application/x-pickle"):
        arr = pickle.loads(body).astype(np.float32)

    else:
        raise ValueError(f"Unsupported content-type: {content_type}")

    # Ensure shape (batch, seq_len, features)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr  # numpy array


def predict_fn(data: np.ndarray, model: nn.Module) -> np.ndarray:
    """
    data : numpy array, shape (batch, seq_len, features)
    """
    B, T, F = data.shape

    # ---- 1) flatten to 2-D (samples, features) for the scaler ----
    flat = data.reshape(-1, F)                    # (B*T, F)
    norm = _normalize(flat, model).astype(np.float32)
    norm = norm.reshape(B, T, F)                  # restore (B, T, F)

    # ---- 2) model forward ----
    tensor = torch.from_numpy(norm).to(next(model.parameters()).device)
    with torch.no_grad():
        raw = model(tensor).cpu().numpy()         # (B, out_features)

    # ---- 3) de-normalize outputs (still 2-D) ----
    denorm = _denormalize(raw, model)             # (B, out_features)
    return denorm


def output_fn(prediction: np.ndarray, accept: str) -> Tuple[str, str]:
    """
    Format the numpy prediction as a JSON string.
    """
    if accept not in ("application/json", "application/json; charset=utf-8"):
        raise ValueError(f"Unsupported accept header: {accept}")
    payload = {"predictions": prediction.tolist()}
    return json.dumps(payload), accept
