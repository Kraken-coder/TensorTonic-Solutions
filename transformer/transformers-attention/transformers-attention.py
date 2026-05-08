import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k = K.size(-1)
    qk = torch.matmul(Q , K.transpose(-2 , -1))/d_k**0.5
    qk = F.softmax(qk , dim = -1 )
    # qk = torch.squeeze(qk)
    return torch.matmul(qk , V)