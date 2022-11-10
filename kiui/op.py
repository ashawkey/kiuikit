import torch
import numpy as np

def normalize(x, dim=-1, eps=1e-20):
    # x: np.ndarray or torch.Tensor
    if isinstance(x, np.ndarray):
        return x / np.sqrt(np.maximum(np.sum(x * x, axis=dim, keepdims=True), eps))
    else:
        return x / torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=True), min=eps))