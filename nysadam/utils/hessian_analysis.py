"""
Hessian Spectrum Analysis
=========================
Author: Hardik Tankaria
Date: May 2026
Description:
    Tools for analyzing the Hessian/Fisher spectrum of neural networks.
    Used for: (1) validating that d_eff is small (justifying low rank m),
    (2) measuring NysAdam's approximation quality, (3) generating
    spectral decay plots for the paper.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Callable


def hessian_vector_product(
    loss_fn: Callable,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    vector: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Compute Hessian-vector product Hv via double backpropagation.

    Args:
        loss_fn: Loss function.
        model: Neural network.
        inputs: Input batch.
        targets: Target batch.
        vector: List of tensors (same shapes as model parameters).

    Returns:
        List of Hv tensors (same shapes as parameters).
    """
    model.zero_grad()
    output = model(inputs)
    loss = loss_fn(output, targets)

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)

    # g^T v
    gv = sum((g * v).sum() for g, v in zip(grads, vector))

    # Hv = d/dw (g^T v)
    hvps = torch.autograd.grad(gv, params)

    return [hvp.detach() for hvp in hvps]


def lanczos_tridiag(
    hvp_fn: Callable,
    dim: int,
    n_iters: int = 100,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Lanczos algorithm to approximate top eigenvalues of the Hessian.

    Args:
        hvp_fn: Function mapping vector → Hv (flattened parameter space).
        dim: Parameter dimension.
        n_iters: Number of Lanczos iterations.
        device: Computation device.
        dtype: Tensor dtype.

    Returns:
        (eigenvalues, eigenvectors) of the tridiagonal matrix.
    """
    alphas = []
    betas = []
    v_prev = torch.zeros(dim, device=device, dtype=dtype)
    v_curr = torch.randn(dim, device=device, dtype=dtype)
    v_curr = v_curr / v_curr.norm()

    for j in range(n_iters):
        w = hvp_fn(v_curr)

        alpha = v_curr.dot(w)
        alphas.append(alpha.item())

        w = w - alpha * v_curr - (betas[-1] * v_prev if betas else 0)

        beta = w.norm()
        if beta < 1e-12:
            break
        betas.append(beta.item())

        v_prev = v_curr
        v_curr = w / beta

    # Build tridiagonal matrix
    k = len(alphas)
    T = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        T[i, i] = alphas[i]
    for i in range(len(betas)):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]

    eigenvalues, eigenvectors = torch.linalg.eigh(T)
    return eigenvalues.float(), eigenvectors.float()


@torch.no_grad()
def estimate_hessian_spectrum(
    model: torch.nn.Module,
    loss_fn: Callable,
    dataloader,
    device: torch.device,
    n_eigenvalues: int = 100,
    n_batches: int = 5,
) -> dict:
    """
    Estimate the top eigenvalues of the Hessian and compute key spectral
    properties including effective dimension.

    Args:
        model: Neural network.
        loss_fn: Loss function.
        dataloader: Data loader.
        device: Device.
        n_eigenvalues: Number of eigenvalues to estimate.
        n_batches: Batches to average over.

    Returns:
        Dictionary with 'eigenvalues', 'effective_dimension', 'spectral_gap',
        'trace_estimate', 'condition_number_estimate'.
    """
    model.eval()

    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)

    # Collect a few batches
    batches = []
    for i, batch in enumerate(dataloader):
        if i >= n_batches:
            break
        batches.append(batch)

    def hvp_fn(v_flat):
        """Hessian-vector product with flattened vector."""
        # Unflatten
        v_list = []
        offset = 0
        for p in params:
            n = p.numel()
            v_list.append(v_flat[offset:offset + n].view_as(p))
            offset += n

        # Average HVP over batches
        hvp_avg = torch.zeros_like(v_flat)
        for batch in batches:
            inputs, targets = batch[0].to(device), batch[1].to(device)
            with torch.enable_grad():
                hvps = hessian_vector_product(loss_fn, model, inputs, targets, v_list)
            flat_hvp = torch.cat([h.view(-1) for h in hvps])
            hvp_avg += flat_hvp
        hvp_avg /= len(batches)
        return hvp_avg

    # Lanczos
    n_iters = min(n_eigenvalues * 2, 300, d)
    eigenvalues, _ = lanczos_tridiag(hvp_fn, d, n_iters, device, params[0].dtype)

    # Sort descending
    eigenvalues = eigenvalues.sort(descending=True).values

    # Effective dimension at various regularization levels
    eff_dims = {}
    for lam in [1e-2, 1e-3, 1e-4, 1e-5]:
        d_eff = (eigenvalues / (eigenvalues.abs() + lam)).sum().item()
        eff_dims[f"d_eff_lambda_{lam}"] = d_eff

    trace_est = eigenvalues.sum().item()
    top = eigenvalues[0].item()
    bottom = eigenvalues[min(n_eigenvalues - 1, len(eigenvalues) - 1)].item()
    cond = abs(top / bottom) if abs(bottom) > 1e-15 else float("inf")

    model.train()

    return {
        "eigenvalues": eigenvalues.cpu().numpy().tolist(),
        "effective_dimensions": eff_dims,
        "spectral_gap": (top - eigenvalues[1].item() if len(eigenvalues) > 1 else 0),
        "trace_estimate": trace_est,
        "top_eigenvalue": top,
        "condition_number_estimate": cond,
        "param_dimension": d,
    }
