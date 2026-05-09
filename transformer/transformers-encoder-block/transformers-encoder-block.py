import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply laye r normalization.
    """
    # Your code here
    mu = np.mean(x , axis = -1 , keepdims = True)
    x = x - mu 
    va = np.var(x , axis = -1 , keepdims = True )
    x = x / np.sqrt(va + eps )
    o = gamma * x + beta
    return o 

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    b , t , d = Q.shape[0] , Q.shape[1] , Q.shape[2]
    dk = W_q.shape[-1] // num_heads
    q = np.matmul(Q , W_q).reshape(b , t , num_heads , -1).transpose(0 , 2 , 1 , 3 ) 
    k = np.matmul(K , W_k).reshape(b , t , num_heads , -1).transpose(0 , 2 , 1 , 3 ) 
    v = np.matmul(V , W_v).reshape(b , t , num_heads , -1).transpose(0 , 2 , 1 , 3 ) 
    attn = np.matmul(q , k.transpose(0 , 1 ,3 , 2 ))/np.sqrt(dk)
    attn = softmax(attn)
    attn = np.matmul(attn , v).transpose(0 , 2 ,1 , 3  ).reshape(b , t , -1 )
    return np.matmul(attn , W_o)
def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    x = np.matmul(x , W1) + b1 
    x = np.maximum(0 , x )
    x = np.matmul(x , W2) + b2
    return x

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here
    mh = multi_head_attention(x , x , x , W_q , W_k , W_v , W_o , num_heads)
    skip = mh + x 
    x = layer_norm(skip , gamma1 , beta1 )
    fcx = feed_forward(x , W1 , b1 ,  W2 , b2)
    x = x + fcx
    output = layer_norm(x , gamma2 , beta2)
    return output