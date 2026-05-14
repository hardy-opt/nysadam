"""
Shampoo: Preconditioned Stochastic Tensor Optimization (Baseline)
=================================================================
Author: Hardik Tankaria
Date: May 2026
Description:
    Simplified implementation of Shampoo for baseline comparison.
    Based on Gupta, Koren & Singer (ICML 2018) and the scalable
    distributed variant by Anil et al. (2021).

    For a matrix parameter W ∈ ℝ^{m×n} with gradient G:
        L_t = β₂ L_{t-1} + (1-β₂) G G^T    (left preconditioner, m×m)
        R_t = β₂ R_{t-1} + (1-β₂) G^T G    (right preconditioner, n×n)
        W_{t+1} = W_t - η L_t^{-1/4} G_t R_t^{-1/4}

    This implementation uses eigendecomposition for matrix p-th roots
    and supports grafting (using Adam's step magnitude with Shampoo's
    direction).

License: MIT
"""

import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Tuple


class Shampoo(Optimizer):
    """
    Shampoo optimizer with optional grafting.

    Args:
        params: Model parameters.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): EMA decay for momentum and
            preconditioner (default: (0.9, 0.999)).
        eps (float): Regularization for matrix roots (default: 1e-12).
        weight_decay (float): Decoupled weight decay (default: 0.01).
        precondition_frequency (int): Steps between preconditioner
            recomputation (default: 100).
        matrix_eps (float): Damping for preconditioner eigenvalues
            (default: 1e-6).
        root_order (int): p-th root exponent, use 4 for standard
            Shampoo -1/4 power (default: 4).
        max_precondition_dim (int): Max dimension for preconditioner
            matrices. Larger layers use Adam fallback (default: 8192).
        graft_type (str): 'none', 'adam', or 'sgd'. Grafting rescales
            Shampoo's direction to match another optimizer's norm
            (default: 'adam').
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-12,
        weight_decay: float = 0.01,
        precondition_frequency: int = 100,
        matrix_eps: float = 1e-6,
        root_order: int = 4,
        max_precondition_dim: int = 8192,
        graft_type: str = "adam",
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            matrix_eps=matrix_eps,
            root_order=root_order,
            max_precondition_dim=max_precondition_dim,
            graft_type=graft_type,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single Shampoo optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            freq = group["precondition_frequency"]
            matrix_eps = group["matrix_eps"]
            root_order = group["root_order"]
            max_dim = group["max_precondition_dim"]
            graft_type = group["graft_type"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(p)

                    # Determine if Shampoo applies (need 2D params, manageable size)
                    use_shampoo = (
                        grad.dim() == 2
                        and max(grad.shape) <= max_dim
                    )
                    state["use_shampoo"] = use_shampoo

                    if use_shampoo:
                        m, n = grad.shape
                        state["L"] = matrix_eps * torch.eye(m, device=p.device, dtype=p.dtype)
                        state["R"] = matrix_eps * torch.eye(n, device=p.device, dtype=p.dtype)
                        state["L_inv_root"] = torch.eye(m, device=p.device, dtype=p.dtype)
                        state["R_inv_root"] = torch.eye(n, device=p.device, dtype=p.dtype)

                    # Adam state for grafting
                    if graft_type == "adam":
                        state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                # Momentum
                momentum = state["momentum"]
                momentum.mul_(beta1).add_(grad, alpha=1 - beta1)
                m_hat = momentum / (1 - beta1 ** step)

                # Adam grafting baseline
                if graft_type == "adam":
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    v_hat = state["exp_avg_sq"] / (1 - beta2 ** step)
                    adam_update = m_hat / (v_hat.sqrt() + 1e-8)
                    adam_norm = adam_update.norm()

                if state.get("use_shampoo", False):
                    G = grad  # [m, n]

                    # Update preconditioner accumulators
                    state["L"].mul_(beta2).add_(G @ G.t(), alpha=1 - beta2)
                    state["R"].mul_(beta2).add_(G.t() @ G, alpha=1 - beta2)

                    # Recompute inverse roots periodically
                    if step % freq == 0:
                        state["L_inv_root"] = self._matrix_inv_root(
                            state["L"], root_order, matrix_eps
                        )
                        state["R_inv_root"] = self._matrix_inv_root(
                            state["R"], root_order, matrix_eps
                        )

                    # Preconditioned update: L^{-1/p} G R^{-1/p}
                    update = state["L_inv_root"] @ m_hat @ state["R_inv_root"]

                    # Grafting: rescale to match Adam's step norm
                    if graft_type == "adam" and adam_norm > 0:
                        shampoo_norm = update.norm()
                        if shampoo_norm > 0:
                            update = update * (adam_norm / shampoo_norm)
                else:
                    # Fall back to Adam for non-2D or oversized params
                    if graft_type == "adam":
                        update = adam_update
                    else:
                        update = m_hat

                # Decoupled weight decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

        return loss

    @staticmethod
    def _matrix_inv_root(
        M: torch.Tensor, root: int, eps: float
    ) -> torch.Tensor:
        """
        Compute M^{-1/root} via eigendecomposition.

        For Shampoo with root=4: computes M^{-1/4}.

        Args:
            M: Symmetric positive semi-definite matrix [n, n].
            root: Root order (e.g., 4 for -1/4 power).
            eps: Damping for small eigenvalues.

        Returns:
            M^{-1/root}, shape [n, n].
        """
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(M)
            # Damp and take inverse root
            eigenvalues = eigenvalues.clamp(min=eps)
            inv_root_eigenvalues = eigenvalues.pow(-1.0 / root)
            return eigenvectors @ torch.diag(inv_root_eigenvalues) @ eigenvectors.t()
        except Exception:
            # Fallback to identity on numerical failure
            return torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
