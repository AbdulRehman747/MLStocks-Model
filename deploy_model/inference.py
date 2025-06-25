# inference.py  (put this in model.tar.gz under code/)
import io, json, pickle, numpy as np, torch, torch.nn as nn
from typing import Any, Dict

########################################
# 1.  Load the model -------------------
########################################
def cpu_pickle_load(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        pkg = pickle.load(f)
    if "model_state_dict" in pkg:
        pkg["model_state_dict"] = {k: v.cpu() for k, v in pkg["model_state_dict"].items()}
    return pkg


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

    def forward(self, x):
        bsz = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, bsz, self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros_like(h0)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(self.dropout(out[:, -1]))


########################################
# 2.  TorchServe entry points ----------
########################################
def model_fn(model_dir: str):
    pkg = cpu_pickle_load(f"{model_dir}/model.pkl")          # your .pkl path
    arch = pkg["model_architecture"]
    model = StockLSTM(**arch).eval()
    model.load_state_dict(pkg["model_state_dict"])
    return model


def input_fn(body, content_type: str):
    """
    Convert the incoming request body to a torch.Tensor.
    """
    if content_type == "application/json":
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        data = json.loads(body)
        arr = np.asarray(data["instances"], dtype=np.float32)

    elif content_type == "application/x-npy":
        arr = np.load(io.BytesIO(body)).astype(np.float32)

    elif content_type in ("application/octet-stream", "application/x-pickle"):
        arr = pickle.loads(body).astype(np.float32)

    else:
        raise ValueError(f"Unsupported content-type: {content_type}")

    # Ensure (batch, seq_len, features) shape
    if arr.ndim == 2:
        arr = arr[None, ...]                # (1, seq_len, features)
    return torch.from_numpy(arr)


def predict_fn(data: torch.Tensor, model):
    with torch.no_grad():
        preds = model(data)
    return preds.cpu().numpy()


def output_fn(prediction, accept: str):
    if accept not in ("application/json", "application/json; charset=utf-8"):
        raise ValueError(f"Unsupported accept header: {accept}")
    return json.dumps({"predictions": prediction.tolist()}), accept
