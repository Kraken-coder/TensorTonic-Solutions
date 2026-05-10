import numpy as np

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    W1 = np.array(W1)
    W2 = np.array(W2)
    x = np.array(x)
    # YOUR CODE HERE
    y = x 
    x = np.matmul(x , W1.transpose(1 , 0 ))
    x = np.maximum( 0 , x )
    x = np.matmul(x , W2.transpose(1 , 0 ))
    x = x + y
    x = np.maximum(0 , x )
    return x 
