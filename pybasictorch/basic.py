from typing import Optional, Tuple

import numpy as np
import torch
import torchvision.transforms

from .utils import dct2d, idct2d, reshape_fortran


def _shrinkageOperator(matrix: torch.Tensor, epsilon: torch.Tensor):
    ZERO = torch.zeros(1, device=matrix.device)
    return torch.minimum(matrix + epsilon, ZERO).add_(torch.maximum(matrix - epsilon, ZERO))


def inexact_alm_rspca_l1(images, weight: Optional[torch.Tensor], lambda_flatfield: float,
                         lambda_darkfield: float, calculate_darkfield: bool, optimization_tolerance: float,
                         max_iterations: int,
                         device, precision=torch.float32):
    if weight is not None and weight.shape != images.shape:
        raise ValueError('weight matrix has different size than input sequence')

    # Initialization and given default variables
    p, q, n = images.shape
    m = p * q
    images = reshape_fortran(images, (m, n)).type(precision)
    if weight is not None:
        weight = reshape_fortran(weight, (m, n)).type(precision)
    else:
        weight = torch.ones_like(images, device=device, dtype=precision)
    svd = torch.linalg.svd(images, full_matrices=False)
    norm_two = svd[1][0]
    Y1 = 0
    ent1 = 1
    ent2 = 10

    A1_hat = torch.zeros_like(images, device=device, dtype=precision)
    A1_coeff = torch.ones((1, images.shape[1]), device=device, dtype=precision)

    E1_hat = torch.zeros_like(images, device=device, dtype=precision)
    W_hat = dct2d(torch.zeros((p, q), device=device, dtype=precision).t())
    mu = 12.5 / norm_two
    mu_bar = mu * 1e7
    rho = 1.5
    d_norm = torch.linalg.norm(images, ord='fro')

    A_offset = torch.zeros((m, 1), device=device, dtype=precision)
    B1_uplimit = torch.min(images)
    B1_offset = 0
    A_inmask = torch.zeros((p, q), device=device, dtype=precision)
    A_inmask[int(np.round(p / 6) - 1): int(np.round(p * 5 / 6)), int(np.round(q / 6) - 1): int(np.round(q * 5 / 6))] = 1
    ZERO = torch.zeros(1, device=device, dtype=precision)
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
        A1_hat = torch.matmul(reshape_fortran(W_idct_hat, (-1, 1)).type(precision), A1_coeff.type(precision)) + A_offset

        temp_W = (images - A1_hat - E1_hat + (1 / mu) * Y1) / ent1
        temp_W = reshape_fortran(temp_W, (p, q, n))
        temp_W = temp_W.mean(dim=2)

        W_hat = W_hat.add_(dct2d(temp_W.t()))
        W_hat = torch.maximum(W_hat - lambda_flatfield / (ent1 * mu), ZERO) + torch.minimum(
            W_hat + lambda_flatfield / (ent1 * mu), ZERO)

        W_idct_hat = idct2d(W_hat.t())
        if A1_coeff.ndim == 1:
            A1_coeff = torch.unsqueeze(A1_coeff, 0)
        if A_offset.ndim == 1:
            A_offset = torch.unsqueeze(A_offset, 1)

        A1_hat = torch.matmul(reshape_fortran(W_idct_hat, (-1, 1)).type(precision), A1_coeff.type(precision)).add_(
            A_offset)

        E1_hat = images - A1_hat + (1 / mu) * Y1 / ent1
        E1_hat = _shrinkageOperator(E1_hat, weight / (ent1 * mu))

        R1 = images - E1_hat
        A1_coeff = R1.mean(dim=0) / R1.mean()
        A1_coeff[A1_coeff < 0] = 0

        if calculate_darkfield:
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

            B1_offset = torch.maximum(B1_offset, ZERO)
            B1_offset = torch.minimum(B1_offset, W_idct_hat.mean() * 1 / B1_uplimit)

            B_offset = B1_offset * reshape_fortran(W_idct_hat, (-1,)) * (-1)

            B_offset = B_offset.add_(torch.ones_like(B_offset) * B1_offset * W_idct_hat.mean())
            A1_offset = torch.mean(R1[:, validA1coeff_idx[0]], dim=1) - torch.mean(
                A1_coeff[validA1coeff_idx[0]]) * reshape_fortran(W_idct_hat, (-1,))
            A1_offset = A1_offset.sub_(A1_offset.mean())

            A_offset = A1_offset - A1_offset.mean() - B_offset

            # smooth A_offset
            W_offset = dct2d(reshape_fortran(A_offset, (p, q)).t())
            W_offset = torch.maximum(W_offset - lambda_darkfield / (ent2 * mu), ZERO) + \
                       torch.minimum(W_offset + lambda_darkfield / (ent2 * mu), ZERO)
            A_offset = idct2d(W_offset.t())
            A_offset = reshape_fortran(A_offset, (-1,))
            # encourage sparse A_offset
            A_offset = torch.maximum(A_offset - lambda_darkfield / (ent2 * mu), ZERO) + \
                       torch.minimum(A_offset + lambda_darkfield / (ent2 * mu), ZERO)
            A_offset = A_offset.add_(B_offset)

        Z1 = images - A1_hat - E1_hat
        Y1 = Y1 + mu * Z1
        mu = torch.minimum(mu * rho, mu_bar)
        # Stop Criterion
        stopCriterion = torch.linalg.norm(Z1, ord='fro') / d_norm

        converged = stopCriterion < optimization_tolerance or iter >= max_iterations

    A_offset = torch.squeeze(A_offset)
    A_offset = A_offset.add_(B1_offset * reshape_fortran(W_idct_hat, (-1,)))

    return A1_hat, E1_hat, A_offset


def basic(image_stack: torch.Tensor, lambda_flatfield: float = 0,
          max_iterations: int = 500,
          optimization_tolerance: float = 1e-6,
          calculate_darkfield: bool = False,
          lambda_darkfield: float = 0,
          working_size: int = 128,
          max_reweight_iterations: int = 10,
          eplson: float = 0.1,
          reweight_tolerance: float = 1e-3, precision: torch.dtype = torch.float32) -> Tuple[
    torch.Tensor, torch.Tensor]:
    """
    Calculate the flatfield and darkfield aberrations
    Args:
        precision: the precision with which to calculate the correction
        image_stack: the image stack. The tensor should be a series of 2D images (i.e. will have a total of 3 dimensions)
        lambda_flatfield: The flatfield regularization parameter. If 0 or negative, will be estimated
        max_iterations:maximum number of iterations in optimisation
        optimization_tolerance: error tolerance
        calculate_darkfield: Whether to calculate darkfield as well
        lambda_darkfield:The darkfield regularization parameter. If 0 or negative, will be estimated
        working_size: the size the images are resized internal
        max_reweight_iterations:
        eplson: reweighting parameter
        reweight_tolerance: reweight tolerance

    Returns: a tuple of the flatfield and darkfield

    """
    device = image_stack.device
    nrows = ncols = working_size
    original_size = image_stack[0].shape

    D = torchvision.transforms.Resize((working_size, working_size))(image_stack).permute(1, 2, 0)

    meanD = D.mean(dim=2)
    meanD = meanD.div_(meanD.mean())
    W_meanD = dct2d(meanD.t())
    if lambda_flatfield == 0:
        lambda_flatfield = W_meanD.abs().sum() / 400 * 0.5
    if lambda_darkfield == 0:
        lambda_darkfield = lambda_flatfield * 0.2
    D = D.sort(dim=2)[0]
    XAoffset = torch.zeros((nrows, ncols), device=device, dtype=precision)
    weight = torch.ones(D.shape, device=device, dtype=precision)

    reweighting_iter = 0
    flag_reweighting = True
    flatfield_last = torch.ones((nrows, ncols), device=device, dtype=precision)
    darkfield_last = torch.randn((nrows, ncols), device=device, dtype=precision)
    while flag_reweighting:
        reweighting_iter += 1

        X_k_A, X_k_E, X_k_Aoffset = inexact_alm_rspca_l1(D, weight=weight, lambda_darkfield=lambda_darkfield,
                                                         lambda_flatfield=lambda_flatfield,
                                                         calculate_darkfield=calculate_darkfield,
                                                         optimization_tolerance=optimization_tolerance,
                                                         max_iterations=max_iterations, device=device,
                                                         precision=precision)

        XA = reshape_fortran(X_k_A, (nrows, ncols, -1))
        XE = reshape_fortran(X_k_E, (nrows, ncols, -1))
        darkfield_current = XAoffset = reshape_fortran(X_k_Aoffset, (nrows, ncols))
        XE_norm = XE.div_(XA.mean(dim=(0, 1)))
        weight = torch.ones_like(XE_norm, device=device, dtype=precision).div_(XE_norm.abs().add_(eplson))
        weight = weight.mul_(weight.numel() / weight.sum())

        temp = XA.mean(dim=2).sub_(XAoffset)
        flatfield_current = temp.div_(temp.mean())

        mad_flatfield = (flatfield_current - flatfield_last).abs().sum().div_(
            flatfield_last.abs().sum())
        temp_diff = (darkfield_current - darkfield_last).abs().sum()
        if temp_diff < 1e-7:
            mad_darkfield = torch.zeros(1, device=device, dtype=precision)

        else:
            mad_darkfield = temp_diff.div_(torch.maximum(darkfield_last.abs().sum(),
                                                         torch.tensor(1e-6, device=device, dtype=precision)))
        flatfield_last = flatfield_current
        darkfield_last = darkfield_current
        if torch.maximum(mad_flatfield,
                         mad_darkfield) <= reweight_tolerance or \
                reweighting_iter >= max_reweight_iterations:
            flag_reweighting = False

    shading = XA.mean(dim=2).sub_(XAoffset)
    flatfield = torch.squeeze(
        torchvision.transforms.Resize((original_size[0], original_size[1]))(torch.unsqueeze(shading, 0)))

    flatfield = flatfield.div_(flatfield.mean())
    if calculate_darkfield:
        darkfield = torch.squeeze(
            torchvision.transforms.Resize((original_size[0], original_size[1]))(torch.unsqueeze(XAoffset, 0)))
    else:
        darkfield = torch.zeros_like(flatfield, device=device, dtype=precision)
    return flatfield, darkfield


def correct_illumination(images, flatfield_img: Optional[torch.Tensor] = None,
                         darkfield_img: Optional[torch.Tensor] = None, **calculate_kwargs) -> torch.Tensor:
    """
    Correct the illumination of an image using the BaSiC algorithm
    Args:
        images: the input images
        flatfield_img: the flatfield image to use for correction. If None provided, then it will be calculated
        darkfield_img: the farkfield image to use for correction
        calculate_kwargs: arguments that will be passed to the basic function (includes an additional 'in_place' argument)

    Returns: a tensor with the corrected image

    """
    in_place = calculate_kwargs.pop("in_place", False)
    if flatfield_img is None:
        flatfield_img, darkfield_img = basic(images, **calculate_kwargs)
    if in_place:
        return images.sub_(darkfield_img).div_(flatfield_img)
    return (images - darkfield_img) / flatfield_img
