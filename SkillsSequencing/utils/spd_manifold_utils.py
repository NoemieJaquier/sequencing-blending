import numpy as np
import scipy as sc
import scipy.linalg as sc_la
import torch


def sqrtm_torch(x):
    """
    This function computes the square root of a matrix.

    Parameters
    ----------
    :param x: positive definite matrix

    Returns
    -------
    :return: sqrtm(x)
    """
    eigenvalues, eigenvectors = torch.symeig(x, eigenvectors=True)

    sqrt_eigenvalues = torch.sqrt(torch.maximum(eigenvalues, torch.zeros_like(eigenvalues)))  # Assume real eigenvalues (first column only)

    return torch.matmul(eigenvectors, torch.matmul(torch.diag_embed(sqrt_eigenvalues), torch.inverse(eigenvectors)))
