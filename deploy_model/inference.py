# deploy_model/inference.py
# -------------------------
"""Custom handler for SageMaker TorchServe endpoint.

• Loads a pickled training-package (model + scalers + config) onto **CPU**  
• Accepts a JSON array-of-records (same schema as training data)  
• Returns 25-minute ahead predictions as JSON
"""

import io, json, pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np


# ---------- utility --------------------------------------------------------- #
def cpu_pickle_load(path: str):
    """Load pickled package and move all tensors to CPU so that
    deserialization works on CPU-only instances."""
    with open(path, "rb") as f:
        pkg = pickle.load(f)
    if "model_state_dict" in pkg:
        pkg["model_state_dict"] = {k: v.cpu() for k, v in pkg["model_state_dict"].items()}
    return pkg


# ---------- model definition ------------------------------------------------ #
class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2,
                 output_size=25, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        bsz = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, bsz, self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out.view(bsz, 5, 5)


# ---------- SageMaker TorchServe hooks ------------------------------------- #
def model_fn(model_dir: str):
    """Called once when the worker starts."""
    pkg   = cpu_pickle_load(f"{model_dir}/model.pkl")
    model = StockLSTM(**pkg["model_architecture"])
    model.load_state_dict(pkg["model_state_dict"])
    model.eval()
    return model, pkg


def input_fn(body, content_type):
    """Convert request payload -> pandas.DataFrame.

    TorchServe hands *bytes*/*bytearray*; Pandas expects str/file-like.
    """
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")

    if not body.strip().startswith("["):
        raise ValueError("Request payload must be a JSON array of records")

    return pd.read_json(io.StringIO(body), orient="records")


def predict_fn(inputs: pd.DataFrame, state):
    """Run prediction on the last *sequence_length* rows."""
    model, pkg      = state
    cfg, scalers    = pkg["config"], pkg["scalers"]
    feats           = cfg["feature_columns"]
    seq_len         = cfg["sequence_length"]
    use_global      = scalers["use_global_scaler"]

    if len(inputs) < seq_len:
        raise ValueError(f"Need ≥{seq_len} rows, got {len(inputs)}")

    X = inputs[feats].values[-seq_len:]

    # -------------------- scale ------------------------------------------------
    if use_global:
        X = scalers["global_scaler"].transform(X)
    else:
        for i, col in enumerate(feats):
            sc = scalers["individual_scalers"].get(col)
            if sc is not None:
                X[:, i] = sc.transform(X[:, i].reshape(-1, 1)).ravel()

    # -------------------- forward pass ----------------------------------------
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32).unsqueeze(0)).squeeze(0).cpu().numpy()

    # -------------------- inverse-scale ---------------------------------------
    if use_global:
        pred = scalers["global_scaler"].inverse_transform(pred)
    else:
        for i, col in enumerate(feats):
            sc = scalers["individual_scalers"].get(col)
            if sc is not None:
                pred[:, i] = sc.inverse_transform(pred[:, i].reshape(-1, 1)).ravel()

    out = pd.DataFrame(pred, columns=feats)
    out["predicted_minute"] = np.arange(1, len(out) + 1)
    return out.to_dict(orient="records")


def output_fn(prediction, _accept):
    """Return JSON string."""
    return json.dumps(prediction), "application/json"
