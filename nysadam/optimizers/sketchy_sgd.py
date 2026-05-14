"""
SketchySGD: Stochastic Optimization via Randomized Curvature Estimates (Baseline)
=================================================================================
Author: Hardik Tankaria
Date: May 2026
Description:
    Simplified SketchySGD for baseline comparison. Based on Frangella,
    Rathore, Zhao & Udell, "SketchySGD," SIAM J. Math. Data Sci., 2024.

    SketchySGD uses a Nyström approximation of the sub-sampled Hessian
    with a SCALAR fill-in (not diagonal like NysAdam). This serves as
    the closest baseline to highlight NysAdam's advantage.

License: MIT
"""

import math
import torch
from torch.optim.optimizer import Optimizer
from typing import Tuple


class SketchySGD(Optimizer):
    """
    SketchySGD: Nyström-preconditioned SGD with scalar regularization.

    The preconditioner is:
        N = Q Q^T + μI  (low-rank + scalar identity)

    Compared to NysAdam:
        - Uses scalar μI instead of diag(v̂+ε) → less adaptive
        - No momentum by default
        - Designed for sub-sampled Hessian → more expensive per update

    Args:
        params: Model parameters.
        lr (float): Learning rate (default: 0.01).
        nys_rank (int): Nyström rank (default: 64).
        mu (float): Scalar regularization (default: 1e-3).
        weight_decay (float): Weight decay (default: 0.01).
        refresh_interval (int): Sketch refresh interval (default: 10).
        momentum (float): SGD momentum (default: 0.9).
        min_dim_for_nystrom (int): Min dim for Nyström (default: 1024).
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        nys_rank: int = 64,
        mu: float = 1e-3,
        weight_decay: float = 0.01,
        refresh_interval: int = 10,
        momentum: float = 0.9,
        min_dim_for_nystrom: int = 1024,
    ):
        defaults = dict(
            lr=lr,
            nys_rank=nys_rank,
            mu=mu,
            weight_decay=weight_decay,
            refresh_interval=refresh_interval,
            momentum=momentum,
            min_dim_for_nystrom=min_dim_for_nystrom,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step_count += 1

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["mu"]
            weight_decay = group["weight_decay"]
            mom = group["momentum"]
            min_dim = group["min_dim_for_nystrom"]
            nys_rank = group["nys_rank"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                d = p.numel()
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["momentum_buf"] = torch.zeros_like(p)
                    use_nys = nys_rank > 0 and d >= min_dim
                    state["use_nystrom"] = use_nys
                    if use_nys:
                        m = min(nys_rank, d)
                        state["nys_Q"] = torch.zeros(d, m, device=p.device, dtype=p.dtype)
                        state["nys_initialized"] = False

                state["step"] += 1

                # Momentum
                buf = state["momentum_buf"]
                buf.mul_(mom).add_(grad)

                # Preconditioned update
                if state.get("use_nystrom", False) and state.get("nys_initialized", False):
                    Q = state["nys_Q"]
                    m_rank = Q.shape[1]
                    g_flat = buf.view(-1)

                    # (Q Q^T + μI)^{-1} g via Woodbury
                    z = g_flat / mu
                    QtDinvQ = Q.t() @ Q / mu
                    A = torch.eye(m_rank, device=Q.device, dtype=Q.dtype) + QtDinvQ
                    Qtz = Q.t() @ z

                    try:
                        L = torch.linalg.cholesky(A)
                        c = torch.cholesky_solve(Qtz.unsqueeze(1), L).squeeze(1)
                    except torch.linalg.LinAlgError:
                        c = torch.linalg.solve(A, Qtz)

                    update = (z - (Q @ c) / mu).view_as(p)
                else:
                    update = buf

                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss

    @torch.no_grad()
    def update_nystrom_sketch(self, per_sample_grads):
        """Update sketch from per-sample gradients (same interface as NysAdam)."""
        for group in self.param_groups:
            nys_rank = group["nys_rank"]
            for p in group["params"]:
                if p not in per_sample_grads:
                    continue
                state = self.state[p]
                if not state.get("use_nystrom", False):
                    continue

                psg = per_sample_grads[p].view(per_sample_grads[p].size(0), -1)
                B, d = psg.shape
                m = min(nys_rank, B, d)

                norms = psg.norm(dim=1)
                _, indices = torch.topk(norms, m)
                G_S = psg[indices].t() / math.sqrt(B)

                try:
                    if d > 4 * m:
                        U, S, _ = torch.svd_lowrank(G_S, q=m)
                    else:
                        U, S, _ = torch.linalg.svd(G_S, full_matrices=False)
                        U, S = U[:, :m], S[:m]
                    state["nys_Q"] = U * S.unsqueeze(0).sqrt()
                    state["nys_initialized"] = True
                except Exception:
                    pass
