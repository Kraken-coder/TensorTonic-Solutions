import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here
    ans = []
    for j in range(seq_length) :
        curr_embedding = [0 ] * d_model
        for i in range(d_model//2) :
            dim = 2*i
            curr_embedding[dim] = np.sin(j/(10000**(2*i/d_model)))
            curr_embedding[dim+1] = np.cos(j/(10000**(2*i/d_model)))
        ans.append(curr_embedding)
    return np.array(ans)