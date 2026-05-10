import numpy as np

def conv_block(x, W1, W2, Ws):
    """
    Returns: np.ndarray with sum of main path output and projected shortcut
    """
    # YOUR CODE HERE
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    Ws = np.array(Ws)
    h = np.matmul(x , W1)
    h = np.maximum( 0 , h)
    z = np.matmul(h , W2)
    # z = np.maximum(0 , z)
    s = np.matmul(x , Ws)
    y = z + s
    y = np.maximum(0 , y )
    return y
