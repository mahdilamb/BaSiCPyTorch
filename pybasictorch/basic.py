from typing import Optional

import numpy as np
import torch
import torchvision.transforms
from tifffile import imsave

from .utils import dct2d, idct2d


def _shrinkageOperator(matrix, epsilon):
    temp1 = matrix - epsilon
    temp1[temp1 < 0] = 0
    temp2 = matrix + epsilon
    temp2[temp2 > 0] = 0
    res = temp1 + temp2
    return res


def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))


def inexact_alm_rspca_l1(images, weight: Optional[torch.Tensor], lambda_flatfield: float,
                         lambda_darkfield: float, darkfield: bool, optimization_tolerance: float, max_iterations: int,
                         device):
    if weight is not None and weight.shape != images.shape:
        raise ValueError('weight matrix has different size than input sequence')

    # if 
    # Initialization and given default variables
    p, q, n = images.shape
    m = p * q
    images = reshape_fortran(images, (m, n))
    if weight is not None:
        weight = reshape_fortran(weight, (m, n))
    else:
        weight = torch.ones_like(images, device=device)
    svd = torch.linalg.svd(images, full_matrices=False)
    norm_two = svd[1][0]
    Y1 = 0
    ent1 = 1
    ent2 = 10

    A1_hat = torch.zeros_like(images, device=device)
    A1_coeff = torch.ones((1, images.shape[1]), device=device)

    E1_hat = torch.zeros_like(images, device=device)
    W_hat = dct2d(torch.zeros((p, q), device=device).t())
    mu = 12.5 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = torch.linalg.norm(images, ord='fro')

    A_offset = torch.zeros((m, 1), device=device)
    B1_uplimit = torch.min(images)
    B1_offset = 0
    A_inmask = torch.zeros((p, q), device=device)
    A_inmask[int(np.round(p / 6) - 1): int(np.round(p * 5 / 6)), int(np.round(q / 6) - 1): int(np.round(q * 5 / 6))] = 1

    # main iteration loop starts
    iter = 0
    converged = False

    while not converged:
        iter += 1
        if A1_coeff.ndim == 1:
            A1_coeff = torch.unsqueeze(A1_coeff, 0)
        if A_offset.ndim == 1:
            A_offset = torch.unsqueeze(A_offset, 1)
        W_idct_hat = idct2d(W_hat.t())
        A1_hat = torch.matmul(reshape_fortran(W_idct_hat, (-1, 1)).float(), A1_coeff.float()) + A_offset

        temp_W = (images - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
        temp_W = reshape_fortran(temp_W, (p, q, n))
        temp_W = temp_W.mean(dim=2)

        W_hat = W_hat + dct2d(temp_W.t())
        W_hat = torch.maximum(W_hat - lambda_flatfield / (ent1 * mu), torch.zeros(1, device=device)) + torch.minimum(
            W_hat + lambda_flatfield / (ent1 * mu), torch.zeros(1, device=device))

        W_idct_hat = idct2d(W_hat.t())
        if A1_coeff.ndim == 1:
            A1_coeff = torch.unsqueeze(A1_coeff, 0)
        if A_offset.ndim == 1:
            A_offset = torch.unsqueeze(A_offset, 1)

        A1_hat = torch.matmul(reshape_fortran(W_idct_hat, (-1, 1)).float(), A1_coeff.float()) + A_offset

        E1_hat = images - A1_hat + (1 / mu) * Y1 / ent1
        E1_hat = _shrinkageOperator(E1_hat, weight / (ent1 * mu))

        R1 = images - E1_hat
        A1_coeff = (R1.mean(dim=0) / R1.mean())
        A1_coeff[A1_coeff < 0] = 0

        if darkfield:
            validA1coeff_idx = torch.where(A1_coeff < 1)
            B1_coeff = torch.mean(
                R1[reshape_fortran(W_idct_hat, (-1,)) > W_idct_hat.mean() - 1e-6][:, validA1coeff_idx[0]],
                0) - torch.mean(R1[reshape_fortran(W_idct_hat, (-1,)) < W_idct_hat.mean() + 1e-6][:,
                                validA1coeff_idx[0]], 0) / R1.mean()
            k = validA1coeff_idx[0].numel()

            temp1 = torch.sum(A1_coeff[validA1coeff_idx[0]] ** 2)
            temp2 = torch.sum(A1_coeff[validA1coeff_idx[0]])
            temp3 = torch.sum(B1_coeff)
            temp4 = torch.sum(A1_coeff[validA1coeff_idx[0]] * B1_coeff)
            temp5 = temp2 * temp3 - temp4 * k
            if temp5 == 0:
                B1_offset = 0
            else:
                B1_offset = (temp1 * temp3 - temp2 * temp4) / temp5
            # limit B1_offset: 0<B1_offset<B1_uplimit

            B1_offset = torch.maximum(B1_offset, torch.zeros(1, device=device))
            B1_offset = torch.minimum(B1_offset, W_idct_hat.mean() * 1 / B1_uplimit)

            B_offset = B1_offset * reshape_fortran(W_idct_hat, (-1,)) * (-1)

            B_offset = B_offset + torch.ones_like(B_offset) * B1_offset * W_idct_hat.mean()
            A1_offset = torch.mean(R1[:, validA1coeff_idx[0]], dim=1) - torch.mean(
                A1_coeff[validA1coeff_idx[0]]) * reshape_fortran(W_idct_hat, (-1,))
            A1_offset = A1_offset - A1_offset.mean()

            A_offset = A1_offset - A1_offset.mean() - B_offset

            # smooth A_offset
            W_offset = dct2d(reshape_fortran(A_offset, (p, q)).t())
            W_offset = torch.maximum(W_offset - lambda_darkfield / (ent2 * mu), torch.zeros(1, device=device)) + \
                       torch.minimum(W_offset + lambda_darkfield / (ent2 * mu), torch.zeros(1, device=device))
            A_offset = idct2d(W_offset.t())
            A_offset = reshape_fortran(A_offset, (-1,))
            # encourage sparse A_offset
            A_offset = torch.maximum(A_offset - lambda_darkfield / (ent2 * mu), torch.zeros(1, device=device)) + \
                       torch.minimum(A_offset + lambda_darkfield / (ent2 * mu), torch.zeros(1, device=device))
            A_offset = A_offset + B_offset

        Z1 = images - A1_hat - E1_hat
        Y1 = Y1 + mu * Z1
        mu = torch.minimum(mu * rho, mu_bar)
        # Stop Criterion
        stopCriterion = torch.linalg.norm(Z1, ord='fro') / d_norm

        converged = converged or stopCriterion < optimization_tolerance or iter >= max_iterations

    A_offset = torch.squeeze(A_offset)
    A_offset = A_offset + B1_offset * reshape_fortran(W_idct_hat, (-1,))

    return A1_hat, E1_hat, A_offset


def basic(images: torch.Tensor, lambda_flatfield: float = 0,
          max_iterations: int = 500,
          optimization_tolerance: float = 1e-6,
          darkfield: bool = False,
          lambda_darkfield: float = 0,
          working_size: int = 128,
          max_reweight_iterations: int = 10,
          eplson: float = 0.1,
          reweight_tolerance: float = 1e-3, ):
    device = images.device
    nrows = ncols = working_size
    _saved_size = images[0].shape

    D = torchvision.transforms.Resize((working_size, working_size))(images).permute(1, 2, 0)

    meanD = D.mean(dim=2)
    meanD = meanD / meanD.mean()
    W_meanD = dct2d(meanD.t())
    if lambda_flatfield == 0:
        lambda_flatfield = torch.sum(torch.abs(W_meanD)) / 400 * 0.5
    if lambda_darkfield == 0:
        lambda_darkfield = lambda_flatfield * 0.2
    D = D.sort(dim=2)[0]
    XAoffset = torch.zeros((nrows, ncols), device=device)
    weight = torch.ones(D.shape, device=device)

    reweighting_iter = 0
    flag_reweighting = True
    flatfield_last = torch.ones((nrows, ncols), device=device)
    darkfield_last = torch.randn((nrows, ncols), device=device)
    while flag_reweighting:
        reweighting_iter += 1

        X_k_A, X_k_E, X_k_Aoffset = inexact_alm_rspca_l1(D, weight=weight, lambda_darkfield=lambda_darkfield,
                                                         lambda_flatfield=lambda_flatfield, darkfield=darkfield,
                                                         optimization_tolerance=optimization_tolerance,
                                                         max_iterations=max_iterations, device=device)

        XA = reshape_fortran(X_k_A, (nrows, ncols, -1))
        XE = reshape_fortran(X_k_E, (nrows, ncols, -1))
        XAoffset = reshape_fortran(X_k_Aoffset, (nrows, ncols))
        XE_norm = XE / XA.mean(dim=(0, 1))
        weight = torch.ones_like(XE_norm, device=device) / (torch.abs(XE_norm) + eplson)
        weight = (weight * weight.numel() / weight.sum())

        temp = XA.mean(dim=2) - XAoffset
        flatfield_current = temp / temp.mean()
        darkfield_current = XAoffset
        mad_flatfield = torch.sum(torch.abs(flatfield_current - flatfield_last)) / torch.sum(torch.abs(flatfield_last))
        temp_diff = torch.sum(torch.abs(darkfield_current - darkfield_last))
        if temp_diff < 1e-7:
            mad_darkfield = torch.zeros(1, device=device)
        else:
            mad_darkfield = temp_diff / torch.maximum(torch.sum(torch.abs(darkfield_last)),
                                                      torch.tensor(1e-6, device=device))
        flatfield_last = flatfield_current
        darkfield_last = darkfield_current
        if torch.maximum(mad_flatfield,
                         mad_darkfield) <= reweight_tolerance or \
                reweighting_iter >= max_reweight_iterations:
            flag_reweighting = False

    shading = XA.mean(dim=2) - XAoffset
    flatfield = torch.squeeze(
        torchvision.transforms.Resize((_saved_size[0], _saved_size[1]))(torch.unsqueeze(shading, 0)))

    flatfield = flatfield / flatfield.mean()
    if darkfield:
        darkfield = torch.squeeze(
            torchvision.transforms.Resize((_saved_size[0], _saved_size[1]))(torch.unsqueeze(XAoffset, 0)))
    else:
        darkfield = torch.zeros_like(flatfield, device=device)
    return flatfield, darkfield


def correct_illumination(images, flatfield: Optional[torch.Tensor] = None,
                         darkfield: Optional[torch.Tensor] = None) -> torch.Tensor:
    if flatfield is None:
        flatfield, darkfield = basic(images)
    return (images - darkfield) / flatfield
