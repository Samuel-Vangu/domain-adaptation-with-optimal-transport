import numpy as np
from ot import sinkhorn



def cost(vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    # vec.shape = (d,)
    # mat.shape = (n, m) 
    # return a ndarray of shape (k,)
    return np.linalg.norm(mat - vec , axis=1)


def cost_matrix(A: np.ndarray, B: np.ndarray):
    """Return the cost matrix between two datasets"""
    # A.shape = (N, d)   → N vectors
    # B.shape = (n, m)
    
    def wrapper(row):
        return cost(row, B)
    
    C = np.apply_along_axis(wrapper, axis=1, arr=A)
    return C*C