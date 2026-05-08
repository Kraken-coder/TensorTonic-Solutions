import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns: Normalized array of same shape as x
    """
    mu = np.mean(x , axis = -1 , keepdims = True )
    x = x - mu 
    va = np.var(x , axis = -1 , keepdims = True )
    x = x/np.sqrt(va + eps)
    o = x *  gamma + beta 
    return o