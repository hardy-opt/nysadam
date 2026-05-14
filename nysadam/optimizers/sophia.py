"""
Sophia-G: Scalable Stochastic Second-order Optimizer (Baseline)
===============================================================
Author: Hardik Tankaria
Date: May 2026
Description:
    Implementation of Sophia-G (Gauss-Newton-Bartlett variant) for
    baseline comparison. Based on Liu et al., "Sophia: A Scalable
    Stochastic Second-order Optimizer for Language Model Pre-training,"
    ICLR 2024. arXiv:2305.14342.

    The key mechanism is element-wise clipping of the preconditioned
    update, acting as an implicit per-coordinate trust region.

    Update rule:
        m_t = β₁ m_{t-1} + (1-β₁) g_t
        h_t = β₂ h_{t-1} + (1-β₂) ĥ_t    (every k steps)
        θ_{t+1} = θ_t - η · clip(m_t / max(h_t, ε), ρ)

License: MIT
"""

import torch
from torch.optim.optimizer import Optimizer
from typing import Tuple


class SophiaG(Optimizer):
    """
    Sophia-G optimizer with Gauss-Newton-Bartlett Hessian estimator.

    Args:
        params: Model parameters.
        lr (float): Learning rate (default: 1e-4).
        betas (Tuple[float, float]): EMA decay rates for momentum and
            Hessian estimate (default: (0.965, 0.99)).
        eps (float): Minimum Hessian estimate value (default: 1e-8).
        weight_decay (float): Decoupled weight decay (default: 0.01).
        rho (float): Clipping threshold for preconditioned update
            (default: 1.0).
        hessian_update_interval (int): Steps between Hessian updates
            (default: 10).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        rho: float = 1.0,
        hessian_update_interval: int = 10,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rho=rho,
            hessian_update_interval=hessian_update_interval,
        )
        super().__init__(params, defaults)
        self._step_count = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single Sophia-G optimization step."""
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
            rho = group["rho"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian_diag"] = torch.ones_like(p) * eps

                state["step"] += 1

                exp_avg = state["exp_avg"]
                h = state["hessian_diag"]

                # Momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # Preconditioned update with clipping
                update = exp_avg / torch.clamp(h, min=eps)
                update.clamp_(-rho, rho)

                # Decoupled weight decay
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

        return loss

    def update_hessian(
        self,
        model: torch.nn.Module,
        loss_fn,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Update diagonal Hessian estimate via Gauss-Newton-Bartlett.

        The GNB estimator uses per-sample gradients:
            ĥ = (1/B) Σᵢ gᵢ²

        This is the diagonal of the empirical Fisher, which is automatically
        PSD and requires no Hessian-vector products.

        Args:
            model: Neural network.
            loss_fn: Loss function accepting (output, target).
            inputs: Input batch [B, ...].
            targets: Target batch [B, ...].
        """
        beta2 = self.defaults["betas"][1]
        B = inputs.size(0)

        # Compute per-sample squared gradients
        model.zero_grad()
        total_sq_grad = {p: torch.zeros_like(p) for p in model.parameters() if p.requires_grad}

        for i in range(B):
            model.zero_grad()
            with torch.enable_grad():
                output = model(inputs[i:i+1])
                loss_i = loss_fn(output, targets[i:i+1])
                loss_i.backward()

            for p in model.parameters():
                if p.grad is not None and p in total_sq_grad:
                    total_sq_grad[p].add_(p.grad.detach() ** 2)

        # Update EMA of Hessian diagonal
        for group in self.param_groups:
            for p in group["params"]:
                if p in total_sq_grad:
                    state = self.state[p]
                    if "hessian_diag" in state:
                        gnb_estimate = total_sq_grad[p] / B
                        state["hessian_diag"].mul_(beta2).add_(
                            gnb_estimate, alpha=1 - beta2
                        )

    def update_hessian_hutchinson(
        self,
        model: torch.nn.Module,
        loss_fn,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        Alternative: Update Hessian diagonal via Hutchinson estimator.
        Uses one Hessian-vector product (Sophia-H variant).
        """
        beta2 = self.defaults["betas"][1]

        model.zero_grad()
        with torch.enable_grad():
            output = model(inputs)
            loss = loss_fn(output, targets)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        params_grads = [
            (p, g) for p, g in zip(model.parameters(), grads) if g is not None
        ]

        # Rademacher vector
        u_vecs = [torch.randint_like(g, 0, 2) * 2.0 - 1.0 for _, g in params_grads]

        # HVP
        gu = sum((g * u).sum() for (_, g), u in zip(params_grads, u_vecs))
        Hu = torch.autograd.grad(gu, [p for p, _ in params_grads])

        for (p, _), u, hu in zip(params_grads, u_vecs, Hu):
            state = self.state.get(p, {})
            if "hessian_diag" in state:
                hess_est = (u * hu).detach()
                state["hessian_diag"].mul_(beta2).add_(hess_est, alpha=1 - beta2)
