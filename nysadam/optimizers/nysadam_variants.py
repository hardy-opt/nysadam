"""
NysAdam Variants
================
Author: Hardik Tankaria
Date: May 2026
Description:
    Variants of NysAdam for different compute/accuracy trade-offs.

    NysAdamLight: Uses gradient accumulation buffers instead of per-sample
        gradients. No per-sample grad computation needed — works as a
        drop-in replacement for AdamW with zero extra infrastructure.

    NysSophia: Replaces Adam's second moment with Sophia's Hutchinson
        Hessian diagonal estimate, giving Nyström cross-parameter +
        Sophia diagonal = best quality curvature at modest cost.

License: MIT
"""

import math
from typing import Dict, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class NysAdamLight(Optimizer):
    """
    NysAdam-Light: Nyström augmentation using gradient history buffer.

    Instead of requiring per-sample gradients, this variant builds the
    Nyström sketch from a rolling buffer of past mini-batch gradients.
    The sketch approximates the empirical gradient covariance:
        Ĉ = (1/K) Σ_{k=1}^K (g_{t-k} - ḡ)(g_{t-k} - ḡ)ᵀ

    This is less accurate than the per-sample Fisher (NysAdam) but
    requires ZERO extra backward passes — making it a true drop-in
    replacement for AdamW.

    Accuracy hierarchy: NysAdam > NysAdamLight >> AdamW

    Args:
        params: Model parameters.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Adam moment decay (default: (0.9, 0.999)).
        eps (float): Numerical stability (default: 1e-8).
        weight_decay (float): Decoupled weight decay (default: 0.01).
        nys_rank (int): Rank of Nyström approximation (default: 32).
        grad_buffer_size (int): Number of past gradients to store for
            sketch construction (default: 128).
        nys_refresh_interval (int): Steps between sketch updates (default: 50).
        nys_beta (float): EMA decay for sketch (default: 0.999).
        clip_threshold (float): Update clipping (default: 1.0).
        reg_lambda (float): Core matrix regularization (default: 1e-4).
        min_dim_for_nystrom (int): Min param dim for Nyström (default: 4096).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        nys_rank: int = 32,
        grad_buffer_size: int = 128,
        nys_refresh_interval: int = 50,
        nys_beta: float = 0.999,
        clip_threshold: float = 1.0,
        reg_lambda: float = 1e-4,
        min_dim_for_nystrom: int = 4096,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            nys_rank=nys_rank,
            grad_buffer_size=grad_buffer_size,
            nys_refresh_interval=nys_refresh_interval,
            nys_beta=nys_beta,
            clip_threshold=clip_threshold,
            reg_lambda=reg_lambda,
            min_dim_for_nystrom=min_dim_for_nystrom,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single NysAdam-Light optimization step."""
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
            clip_threshold = group["clip_threshold"]
            nys_rank = group["nys_rank"]
            buf_size = group["grad_buffer_size"]
            refresh_k = group["nys_refresh_interval"]
            nys_beta = group["nys_beta"]
            reg_lambda = group["reg_lambda"]
            min_dim = group["min_dim_for_nystrom"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                d = p.numel()
                state = self.state[p]

                # ---- Initialize state ----
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                    # Gradient buffer for sketch construction.
                    # Memory guard: for large parameters (e.g. vocab embeddings
                    # with d=12.8M), the buffer alone would be buf_size*d*4 bytes
                    # = 6.4 GB. Skip Nystrom for such layers.
                    buffer_bytes = buf_size * d * p.element_size()
                    max_buffer_bytes = 256 * 1024 * 1024  # 256 MB per parameter
                    use_nys = (nys_rank > 0) and (d >= min_dim) and (buffer_bytes <= max_buffer_bytes)
                    state["use_nystrom"] = use_nys

                    if use_nys:
                        m = min(nys_rank, d)
                        state["grad_buffer"] = torch.zeros(
                            buf_size, d, device=p.device, dtype=p.dtype
                        )
                        state["buf_idx"] = 0
                        state["buf_filled"] = False
                        state["nys_Q"] = torch.zeros(d, m, device=p.device, dtype=p.dtype)
                        state["nys_Phi"] = reg_lambda * torch.eye(
                            m, device=p.device, dtype=p.dtype
                        )
                        state["nys_initialized"] = False

                state["step"] += 1
                step = state["step"]

                # ---- Adam moments ----
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # ---- Accumulate gradient in buffer ----
                if state.get("use_nystrom", False):
                    buf = state["grad_buffer"]
                    idx = state["buf_idx"]
                    buf[idx] = grad.view(-1)
                    state["buf_idx"] = (idx + 1) % buf_size
                    if idx + 1 >= buf_size:
                        state["buf_filled"] = True

                    # ---- Refresh Nyström sketch every k steps ----
                    if step % refresh_k == 0 and (state["buf_filled"] or idx >= nys_rank):
                        n_available = buf_size if state["buf_filled"] else idx + 1
                        grads_available = buf[:n_available]  # [n_avail, d]
                        m_eff = min(nys_rank, n_available, d)

                        # Select top-norm gradients as landmarks
                        norms = grads_available.norm(dim=1)
                        _, top_idx = torch.topk(norms, min(m_eff, n_available))
                        G_S = grads_available[top_idx].t()  # [d, m_eff]

                        # EMA update of sketch
                        Q_old = state["nys_Q"]
                        scale_new = math.sqrt(1 - nys_beta) / math.sqrt(n_available)

                        if state["nys_initialized"]:
                            Q_combined = torch.cat(
                                [math.sqrt(nys_beta) * Q_old, scale_new * G_S], dim=1
                            )
                        else:
                            Q_combined = scale_new * G_S
                            state["nys_initialized"] = True

                        # Truncated SVD
                        try:
                            if Q_combined.shape[0] > 4 * m_eff:
                                U, S, _ = torch.svd_lowrank(Q_combined, q=m_eff)
                            else:
                                U, S, _ = torch.linalg.svd(Q_combined, full_matrices=False)
                                U = U[:, :m_eff]
                                S = S[:m_eff]
                            Q_new = U * S.unsqueeze(0).sqrt()
                            if torch.isfinite(Q_new).all():
                                state["nys_Q"] = Q_new
                        except Exception:
                            pass  # Keep old sketch on SVD failure

                        Q = state["nys_Q"]
                        state["nys_Phi"] = Q.t() @ Q + reg_lambda * torch.eye(
                            Q.shape[1], device=Q.device, dtype=Q.dtype
                        )

                # ---- Compute update ----
                if state.get("use_nystrom", False) and state.get("nys_initialized", False):
                    update = self._woodbury_update(
                        m_hat.view(-1), v_hat.view(-1),
                        state["nys_Q"], state["nys_Phi"], eps,
                    ).view_as(p)
                else:
                    update = m_hat / (v_hat.sqrt() + eps)

                # ---- Clip + weight decay + update ----
                if clip_threshold < float("inf"):
                    update.clamp_(-clip_threshold, clip_threshold)
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss

    @staticmethod
    def _woodbury_update(m_hat, v_hat, Q, Phi, eps):
        """Woodbury inverse application: (Q Φ⁻¹ Qᵀ + D²)⁻¹ m̂."""
        d_inv_sq = 1.0 / (v_hat + eps)
        z = d_inv_sq * m_hat
        D_inv_Q = Q * d_inv_sq.unsqueeze(1)
        A = Phi + Q.t() @ D_inv_Q

        Qt_z = Q.t() @ z
        try:
            L = torch.linalg.cholesky(A)
            c = torch.cholesky_solve(Qt_z.unsqueeze(1), L).squeeze(1)
        except torch.linalg.LinAlgError:
            c = torch.linalg.solve(A, Qt_z)

        return z - D_inv_Q @ c


class NysSophia(Optimizer):
    """
    NysSophia: Nyström cross-parameter curvature + Sophia's Hessian diagonal.

    Replaces Adam's gradient-second-moment diagonal with Sophia's Hutchinson
    estimator of the diagonal Hessian. This gives higher-quality diagonal
    curvature information at the cost of one Hessian-vector product every
    k_hessian steps.

    The preconditioner is:
        N_t = Q_t Φ⁻¹ Q_tᵀ + diag(ĥ_t + ε)

    where ĥ_t is the EMA of diagonal Hessian estimates (Hutchinson or GNB).

    Args:
        params: Model parameters.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Moment decay (default: (0.9, 0.999)).
        eps (float): Numerical stability (default: 1e-8).
        weight_decay (float): Decoupled weight decay (default: 0.01).
        nys_rank (int): Nyström rank (default: 64).
        nys_refresh_interval (int): Sketch refresh interval (default: 10).
        nys_beta (float): Sketch EMA decay (default: 0.999).
        hessian_refresh_interval (int): Steps between Hessian diagonal
            updates (default: 10).
        clip_threshold (float): Update clipping (default: 1.0).
        reg_lambda (float): Core matrix regularization (default: 1e-4).
        min_dim_for_nystrom (int): Min dim for Nyström (default: 1024).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        nys_rank: int = 64,
        nys_refresh_interval: int = 10,
        nys_beta: float = 0.999,
        hessian_refresh_interval: int = 10,
        clip_threshold: float = 1.0,
        reg_lambda: float = 1e-4,
        min_dim_for_nystrom: int = 1024,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            nys_rank=nys_rank,
            nys_refresh_interval=nys_refresh_interval,
            nys_beta=nys_beta,
            hessian_refresh_interval=hessian_refresh_interval,
            clip_threshold=clip_threshold,
            reg_lambda=reg_lambda,
            min_dim_for_nystrom=min_dim_for_nystrom,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single NysSophia optimization step."""
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
            clip_threshold = group["clip_threshold"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian_diag"] = torch.ones_like(p) * eps
                    d = p.numel()
                    m = group["nys_rank"] if d >= group["min_dim_for_nystrom"] else 0
                    state["nys_effective_rank"] = m
                    if m > 0:
                        state["nys_Q"] = torch.zeros(d, m, device=p.device, dtype=p.dtype)
                        state["nys_Phi"] = group["reg_lambda"] * torch.eye(
                            m, device=p.device, dtype=p.dtype
                        )
                        state["nys_initialized"] = False

                state["step"] += 1
                step = state["step"]

                # First moment
                exp_avg = state["exp_avg"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                m_hat = exp_avg / (1 - beta1 ** step)

                # Use Hessian diagonal (updated externally via update_hessian_diagonal)
                h_diag = state["hessian_diag"]

                # Preconditioned update
                m_eff = state.get("nys_effective_rank", 0)
                if m_eff > 0 and state.get("nys_initialized", False):
                    d_inv_sq = 1.0 / (h_diag.view(-1).abs() + eps)
                    m_flat = m_hat.view(-1)
                    z = d_inv_sq * m_flat
                    Q = state["nys_Q"]
                    Phi = state["nys_Phi"]
                    D_inv_Q = Q * d_inv_sq.unsqueeze(1)
                    A = Phi + Q.t() @ D_inv_Q
                    Qt_z = Q.t() @ z
                    try:
                        L = torch.linalg.cholesky(A)
                        c = torch.cholesky_solve(Qt_z.unsqueeze(1), L).squeeze(1)
                    except torch.linalg.LinAlgError:
                        c = torch.linalg.solve(A, Qt_z)
                    update = (z - D_inv_Q @ c).view_as(p)
                else:
                    update = m_hat / (h_diag.abs() + eps)

                if clip_threshold < float("inf"):
                    update.clamp_(-clip_threshold, clip_threshold)
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)
                p.add_(update, alpha=-lr)

        return loss

    def update_hessian_diagonal(
        self,
        model: torch.nn.Module,
        loss_fn,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Update the diagonal Hessian estimate via Hutchinson's method.
        Call every `hessian_refresh_interval` steps.

        Uses one Hessian-vector product with a Rademacher random vector.
        """
        beta2 = self.defaults["betas"][1]

        model.zero_grad()
        with torch.enable_grad():
            output = model(inputs)
            loss = loss_fn(output, targets)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Hutchinson estimator: h ≈ u ⊙ (H u) for u ~ Rademacher
        params_with_grad = [
            (p, g) for p, g in zip(model.parameters(), grads) if g is not None
        ]

        u_vectors = [torch.randint_like(g, 0, 2) * 2.0 - 1.0 for _, g in params_with_grad]

        # Hessian-vector product: Hu = d/dw (g^T u)
        gu = sum((g * u).sum() for (_, g), u in zip(params_with_grad, u_vectors))
        Hu = torch.autograd.grad(gu, [p for p, _ in params_with_grad])

        for (p, _), u, hu in zip(params_with_grad, u_vectors, Hu):
            if p in self.state:
                state = self.state[p]
                if "hessian_diag" in state:
                    hess_est = (u * hu).detach()
                    state["hessian_diag"].mul_(beta2).add_(
                        hess_est, alpha=1 - beta2
                    )

    def update_nystrom_sketch(
        self,
        per_sample_grads: Dict[torch.nn.Parameter, torch.Tensor],
    ) -> None:
        """Update Nyström sketch (identical logic to NysAdam)."""
        for group in self.param_groups:
            nys_beta = group["nys_beta"]
            reg_lambda = group["reg_lambda"]

            for p in group["params"]:
                if p not in per_sample_grads:
                    continue
                state = self.state[p]
                m = state.get("nys_effective_rank", 0)
                if m == 0:
                    continue

                psg = per_sample_grads[p].view(per_sample_grads[p].size(0), -1)
                B = psg.size(0)

                norms = psg.norm(dim=1)
                _, indices = torch.topk(norms, min(m, B))
                G_S = psg[indices].t()

                Q_old = state["nys_Q"]
                scale_new = math.sqrt(1 - nys_beta) / math.sqrt(B)

                if state.get("nys_initialized", False):
                    Q_combined = torch.cat(
                        [math.sqrt(nys_beta) * Q_old, scale_new * G_S], dim=1
                    )
                else:
                    Q_combined = scale_new * G_S
                    state["nys_initialized"] = True

                try:
                    if Q_combined.shape[0] > 4 * m:
                        U, S, _ = torch.svd_lowrank(Q_combined, q=m)
                    else:
                        U, S, _ = torch.linalg.svd(Q_combined, full_matrices=False)
                        U, S = U[:, :m], S[:m]
                    Q_new = U * S.unsqueeze(0).sqrt()
                    if torch.isfinite(Q_new).all():
                        state["nys_Q"] = Q_new
                except Exception:
                    pass

                Q = state["nys_Q"]
                state["nys_Phi"] = Q.t() @ Q + reg_lambda * torch.eye(
                    Q.shape[1], device=Q.device, dtype=Q.dtype
                )
