import numpy as np
from ot import sinkhorn





def cost_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cost matrix with squared Euclidean distance"""
    # A: (N, d), B: (n, d)
    diff = A[:, None, :] - B[None, :, :]   # (N, n, d)
    C = np.linalg.norm(diff, axis=-1)      # (N, n)
    return C ** 2


