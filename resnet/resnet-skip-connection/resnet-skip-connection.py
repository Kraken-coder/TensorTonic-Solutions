import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # grad = np.array(gradients_F)
    # ide = np.ones(grad.shape)
    # grad = grad + ide 
    # g = np.array(gradients_F[0])
    for i in range(0 , len(gradients_F) , 1):
        f = np.array(gradients_F[i])
        ones = np.identity(f.shape[0])
        f = ones + f
        x = np.matmul(x , f)
    return x 


def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # ide = np.ones(grad.shape)
    # grad = grad + ide 
    # g = np.array(gradients_F[0])
    for i in range(len(gradients_F)):
        f = np.array(gradients_F[i])
        # ones = np.identity(f.shape)
        # f = ones + f
        x = np.matmul(x , f)
    return x
