"""
Tensor Serializer
Converts PyTorch tensors to JSON-safe structures for API responses.
"""

import torch
import numpy as np
from api.schemas.responses import TensorData


def tensor_to_json(t: torch.Tensor, max_elements: int = 50_000) -> TensorData:
    """
    Convert a PyTorch tensor to a TensorData schema.

    For large tensors (e.g. weight matrices), values are truncated to
    `max_elements` to keep response sizes reasonable.
    """
    shape = list(t.shape)
    dtype_str = str(t.dtype).replace("torch.", "")

    # Detach and move to CPU
    arr = t.detach().cpu()

    # Truncate very large tensors
    numel = arr.numel()
    if numel > max_elements:
        # For 2D matrices, take top-left corner
        if arr.ndim == 2:
            max_rows = min(arr.shape[0], int(max_elements ** 0.5))
            max_cols = min(arr.shape[1], int(max_elements ** 0.5))
            arr = arr[:max_rows, :max_cols]
        else:
            arr = arr.flatten()[:max_elements].reshape(-1)

    # Convert to nested Python lists (JSON-serializable)
    data = arr.float().numpy().tolist()

    return TensorData(shape=shape, data=data, dtype=dtype_str)


def serialize_internals(raw: dict) -> dict[str, TensorData]:
    """
    Convert a dict of tensors (from model.get_internals()) to a dict of
    TensorData schemas. Non-tensor values are skipped.
    """
    result = {}
    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            result[key] = tensor_to_json(value)
        elif isinstance(value, np.ndarray):
            result[key] = tensor_to_json(torch.from_numpy(value))
        # Skip strings, metadata, etc.
    return result
