import numpy as np

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    # YOUR CODE HERE
    s = x 
    x = np.matmul( x , W1 )
    x = np.maximum( 0 , x )
    x = np.matmul(x , W2 )
    x = np.maximum( 0 , x )
    x = np.matmul(x , W3 )
    
    x = x + np.matmul(s , Ws)
    x = np.maximum( 0 , x)
    return x
