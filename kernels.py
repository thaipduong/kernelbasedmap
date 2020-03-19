import numpy as np

def linear_kernel(x1, x2):
    return np.inner(x1, x2) + 1

def rbf_kernel(x1, x2, gamma = 2):
    k = np.exp(-gamma*(np.linalg.norm(x1 - x2)**2))
    return k