import torch
from typing import Any

def recursive_to_device(data: Any, device: torch.device) -> Any:
    """Recursively moves tensors in nested structures (list, dict, tuple) to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: recursive_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [recursive_to_device(v, device) for v in data]
    elif isinstance(data, tuple):
        return tuple(recursive_to_device(v, device) for v in data)
    return data
