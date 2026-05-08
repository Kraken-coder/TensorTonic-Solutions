import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    b , t , d = Q.shape[0] , Q.shape[1] , Q.shape[2]
    dk = d//num_heads
    q = np.matmul(Q , W_q).reshape(b , t , num_heads , -1 ).transpose(0 , 2 , 1 , 3 )
    k = np.matmul(K , W_k).reshape(b , t , num_heads , -1 ).transpose(0 , 2 , 1 , 3 )
    v = np.matmul(V , W_v).reshape(b , t , num_heads , -1 ).transpose(0 , 2 , 1 , 3 )
    # Your code here
    attn = np.matmul(q , k.transpose(0 , 1, 3 , 2 ))/np.sqrt(dk)
    attn = softmax(attn)
    attn = np.matmul(attn , v)
    attn = attn.transpose(0, 2, 1, 3)
    attn = attn.reshape(b , t , -1)
    
    return np.matmul(attn, W_o) 