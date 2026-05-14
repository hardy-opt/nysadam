"""
Nyström Mathematical Utilities
==============================
Author: Hardik Tankaria
Date: May 2026
Description:
    Core mathematical operations for Nyström-based preconditioning:
    - Woodbury identity application
    - Effective dimension computation
    - Condition number estimation
    - Spectral analysis of Nyström approximation quality
"""

import torch
import math
from typing import Tuple, Optional


def woodbury_solve(
    Q: torch.Tensor,
    Phi: torch.Tensor,
    D_diag: torch.Tensor,
    rhs: torch.Tensor,
) -> torch.Tensor:
    """
    Solve (Q Φ⁻¹ Qᵀ + D)x = rhs via Woodbury identity.

    (Q Φ⁻¹ Qᵀ + D)⁻¹ = D⁻¹ - D⁻¹ Q (Φ + Qᵀ D⁻¹ Q)⁻¹ Qᵀ D⁻¹

    Args:
        Q: Nyström sketch [d, m].
        Phi: Core matrix [m, m] (SPD).
        D_diag: Diagonal of D [d] (positive).
        rhs: Right-hand side [d].

    Returns:
        Solution x [d].
    """
    D_inv = 1.0 / D_diag
    z = D_inv * rhs

    D_inv_Q = Q * D_inv.unsqueeze(1)
    A = Phi + Q.t() @ D_inv_Q

    Qt_z = Q.t() @ z

    try:
        L = torch.linalg.cholesky(A)
        c = torch.cholesky_solve(Qt_z.unsqueeze(1), L).squeeze(1)
    except torch.linalg.LinAlgError:
        c = torch.linalg.solve(A, Qt_z)

    return z - D_inv_Q @ c


def compute_effective_dimension(
    eigenvalues: torch.Tensor,
    reg_lambda: float,
) -> float:
    """
    Compute the effective dimension d_eff(λ) = Σ λᵢ/(λᵢ + λ).

    This determines the minimum Nyström rank needed for near-optimal
    preconditioning (Frangella, Tropp & Udell, 2023).

    Args:
        eigenvalues: Eigenvalues of the curvature matrix [d].
        reg_lambda: Regularization parameter.

    Returns:
        Effective dimension (float).
    """
    return (eigenvalues / (eigenvalues + reg_lambda)).sum().item()


def nystrom_approximation_error(
    A: torch.Tensor,
    Q: torch.Tensor,
    Phi: torch.Tensor,
) -> Tuple[float, float]:
    """
    Compute the spectral and Frobenius norm errors of the Nyström
    approximation A ≈ Q Φ⁻¹ Qᵀ.

    Args:
        A: True matrix [d, d].
        Q: Nyström sketch [d, m].
        Phi: Core matrix [m, m].

    Returns:
        (spectral_error, frobenius_error) as floats.
    """
    Phi_inv = torch.linalg.inv(Phi)
    A_nys = Q @ Phi_inv @ Q.t()
    diff = A - A_nys

    spectral_err = torch.linalg.norm(diff, ord=2).item()
    frobenius_err = torch.linalg.norm(diff, ord="fro").item()

    return spectral_err, frobenius_err


def estimate_condition_number_nystrom(
    Q: torch.Tensor,
    Phi: torch.Tensor,
    D_diag: torch.Tensor,
    n_probes: int = 20,
) -> float:
    """
    Estimate the condition number of the Nyström preconditioned system
    using random probing (Lanczos-style).

    Args:
        Q: Nyström sketch [d, m].
        Phi: Core matrix [m, m].
        D_diag: Adam diagonal [d].
        n_probes: Number of random probe vectors.

    Returns:
        Estimated condition number.
    """
    d = Q.shape[0]

    def matvec(v):
        """Apply N = Q Φ⁻¹ Qᵀ + D to vector v."""
        Phi_inv_Qt_v = torch.linalg.solve(Phi, Q.t() @ v)
        return Q @ Phi_inv_Qt_v + D_diag * v

    # Power iteration for largest eigenvalue
    v = torch.randn(d, device=Q.device, dtype=Q.dtype)
    v = v / v.norm()
    for _ in range(50):
        w = matvec(v)
        lambda_max = v.dot(w)
        v = w / w.norm()

    # Inverse power iteration for smallest eigenvalue
    v = torch.randn(d, device=Q.device, dtype=Q.dtype)
    v = v / v.norm()
    for _ in range(50):
        w = woodbury_solve(Q, Phi, D_diag, v)
        v = w / w.norm()
    lambda_min = 1.0 / v.dot(matvec(v))

    cond = abs(lambda_max / lambda_min) if abs(lambda_min) > 1e-15 else float("inf")
    return cond


def truncated_svd_stable(
    M: torch.Tensor,
    rank: int,
    n_oversamples: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Numerically stable truncated SVD returning U, S for top `rank` components.

    Uses randomized SVD with oversampling for large matrices.

    Args:
        M: Input matrix [d, k].
        rank: Target rank.
        n_oversamples: Extra samples for randomized SVD accuracy.

    Returns:
        U [d, rank], S [rank].
    """
    d, k = M.shape
    rank = min(rank, k, d)

    if d > 4 * rank:
        try:
            U, S, _ = torch.svd_lowrank(M, q=rank, niter=4)
            return U[:, :rank], S[:rank]
        except Exception:
            pass

    try:
        U, S, _ = torch.linalg.svd(M, full_matrices=False)
        return U[:, :rank], S[:rank]
    except Exception:
        gram = M @ M.t()
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        idx = torch.argsort(eigenvalues, descending=True)[:rank]
        U = eigenvectors[:, idx]
        S = eigenvalues[idx].clamp(min=0).sqrt()
        return U, S


def compute_leverage_scores(
    G: torch.Tensor,
    Q: torch.Tensor,
    Phi: torch.Tensor,
    reg_lambda: float = 1e-4,
) -> torch.Tensor:
    """
    Compute approximate leverage scores for landmark selection.

    ℓᵢ ≈ gᵢᵀ (Q Φ⁻¹ Qᵀ + λI)⁻¹ gᵢ

    Higher leverage scores indicate more "informative" gradients.

    Args:
        G: Per-sample gradients [B, d].
        Q: Current Nyström sketch [d, m].
        Phi: Core matrix [m, m].
        reg_lambda: Regularization.

    Returns:
        Leverage scores [B].
    """
    B, d = G.shape
    scores = torch.zeros(B, device=G.device, dtype=G.dtype)

    D_diag = torch.full((d,), reg_lambda, device=G.device, dtype=G.dtype)

    for i in range(B):
        gi = G[i]
        solved = woodbury_solve(Q, Phi, D_diag, gi)
        scores[i] = gi.dot(solved)

    return scores
