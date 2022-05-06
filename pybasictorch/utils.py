import os
from fnmatch import fnmatch

import numpy as np
import torch
from tifffile import imread
from torch_dct import dct, idct


def load_data(directory: str, pattern: str = "*.tif*") -> torch.Tensor:
    """
    Load a dataset into a tensor. Assumes that the input files
    Args:
        directory: the directory to search for images
        pattern: the pattern the images need to match

    Returns: a torch tensor

    """
    return torch.tensor(np.asarray([imread(image) for image in sorted(
        os.path.join(directory, file) for file in os.listdir(directory) if fnmatch(file, pattern))]) * 1.0)


def dct2d(mtrx: torch.Tensor):
    """
    Calculates 2D discrete cosine transform.

    Parameters
    ----------
    mtrx
        Input matrix.

    Returns
    -------
    Discrete cosine transform of the input matrix.
    """
    # Check if input object is 2D.
    if mtrx.ndim != 2:
        raise ValueError("Passed object should be a matrix or a numpy array with dimension of two.")

    return dct(dct(mtrx.t(), norm='ortho').t(), norm='ortho')


def idct2d(mtrx: torch.Tensor):
    """
    Calculates 2D inverse discrete cosine transform.

    Parameters
    ----------
    mtrx
        Input matrix.

    Returns
    -------
    Inverse of discrete cosine transform of the input matrix.
    """

    # Check if input object is 2D.
    if mtrx.ndim != 2:
        raise ValueError("Passed object should be a matrix or a numpy array with dimension of two.")

    return idct(idct(mtrx.t(), norm='ortho').t(), norm='ortho')
