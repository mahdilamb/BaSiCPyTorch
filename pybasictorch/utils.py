import os
from fnmatch import fnmatch
from typing import Callable

import numpy as np
import torch


def dct(x, norm=None):
    """
    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    @author git+https://github.com/jbojar/torch-dct@pytorch-1.9.0-compatibility
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
:author git+https://github.com/jbojar/torch-dct@pytorch-1.9.0-compatibility

    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, :N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, :N // 2]

    return x.view(*x_shape)


def load_data(directory: str, pattern: str = "*.tif*", reader: Callable[[str], np.ndarray] = None) -> torch.Tensor:
    if reader is None:
        from tifffile import imread
        reader = imread

    """
    Load a dataset into a tensor. Assumes that the input files
    Args:
        directory: the directory to search for images
        pattern: the pattern the images need to match

    Returns: a torch tensor

    """
    return torch.tensor(np.asarray([reader(image) for image in sorted(
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
        raise ValueError(f"Tensor must be 2D. Matrix is {mtrx.ndim}D")

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
        raise ValueError(f"Tensor must be 2D. Matrix is {mtrx.ndim}D")

    return idct(idct(mtrx.t(), norm='ortho').t(), norm='ortho')


def reshape_fortran(x, shape):
    """
    Reshape a tensor with fortran ordering
    Original code posted on StackOverflow https://stackoverflow.com/a/63964246/979591
    Usage licenced by CC BY-SA 4.0
    Args:
        x: the input tensor
        shape: the shape of the required shape

    Returns: the reshaped tensor

    """
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
