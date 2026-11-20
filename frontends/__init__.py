from pathlib import Path
from ..ir import Graph


def load_graph(path: str, dynamic_dim: int = 1) -> Graph:
    ext = Path(path).suffix.lower()
    if ext == ".onnx":
        from .onnx_frontend import from_onnx
        return from_onnx(path, dynamic_dim)
    if ext == ".tflite":
        from .tflite_frontend import from_tflite
        return from_tflite(path, dynamic_dim)
    if ext == ".csv":
        from .csv_frontend import from_csv
        return from_csv(path)
    raise ValueError(f"Unsupported model format '{ext}' (use .onnx, .tflite or .csv)")
