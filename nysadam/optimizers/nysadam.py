"""
NysAdam: Nyström-Augmented Adaptive Moment Estimation
=====================================================
Author: Hardik Tankaria
Date: May 2026
Description:
    Core implementation of the NysAdam optimizer. NysAdam augments Adam's
    per-parameter adaptive learning rates with a rank-m Nyström approximation
    of cross-parameter curvature from the empirical Fisher information matrix.

    The preconditioner is:
        N_t = Q_t Φ⁻¹ Q_tᵀ + diag(v̂_t + ε)

    where Q_t ∈ ℝ^{d_ℓ × m} is the Nyström sketch (per layer), Φ = Q'Q + λI
    is the m×m core matrix, and v̂_t is Adam's bias-corrected second moment.

    Inversion is performed via the Sherman-Morrison-Woodbury identity:
        N⁻¹ x = D⁻¹ x - D⁻¹ Q A⁻¹ Qᵀ D⁻¹ x
    where D = diag(v̂_t + ε), A = Φ + Qᵀ D⁻¹ Q  ∈ ℝ^{m×m}.

    When m=0, NysAdam reduces exactly to AdamW.

References:
    [1] Frangella, Tropp, Udell. "Randomized Nyström Preconditioning." 2023.
    [2] Liu et al. "Sophia: A Scalable Stochastic Second-order Optimizer." 2023.
    [3] Loshchilov & Hutter. "Decoupled Weight Decay Regularization." 2019.
    [4] Kingma & Ba. "Adam: A Method for Stochastic Optimization." 2015.

License: MIT
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
from torch.optim.optimizer import Optimizer


class NysAdam(Optimizer):
    """
    NysAdam: Nyström-Augmented Adaptive Moment Estimation.

    Combines Adam's diagonal second-moment preconditioning with a low-rank
    Nyström approximation of the empirical Fisher's top eigendirections.
    Smoothly interpolates between AdamW (nys_rank=0) and the natural
    gradient (nys_rank → layer_dim).

    Args:
        params (iterable): Iterable of parameters to optimize or dicts
            defining parameter groups.
        lr (float): Learning rate (default: 1e-3).
        betas (Tuple[float, float]): Coefficients for computing running
            averages of gradient and its square (default: (0.9, 0.999)).
        eps (float): Term added to denominator for numerical stability
            (default: 1e-8).
        weight_decay (float): Decoupled weight decay coefficient
            (default: 0.01).
        nys_rank (int): Rank m of the Nyström approximation. Set to 0
            to recover exact AdamW behavior (default: 64).
        nys_refresh_interval (int): Number of steps between Nyström
            sketch updates (default: 10).
        nys_beta (float): EMA decay rate for Nyström sketch factors
            (default: 0.999).
        clip_threshold (float): Element-wise clipping threshold for
            the preconditioned update. Set to float('inf') to disable
            (default: 1.0).
        reg_lambda (float): Regularization parameter for the Nyström
            core matrix Φ = Q'Q + reg_lambda * I (default: 1e-4).
        landmark_selection (str): Strategy for selecting landmark
            gradients: 'topk_norm', 'random', or 'leverage'
            (default: 'topk_norm').
        min_dim_for_nystrom (int): Minimum parameter dimension to apply
            Nyström. Smaller layers use pure Adam (default: 1024).

    Example:
        >>> optimizer = NysAdam(model.parameters(), lr=1e-3, nys_rank=64)
        >>> for inputs, targets in dataloader:
        ...     # Forward pass with per-sample gradient support
        ...     loss = loss_fn(model(inputs), targets)
        ...     optimizer.zero_grad()
        ...     loss.backward()
        ...     optimizer.step()
        ...     # Feed per-sample gradients for Nyström update
        ...     if step % optimizer.defaults['nys_refresh_interval'] == 0:
        ...         per_sample_grads = compute_per_sample_grads(model, inputs, targets)
        ...         optimizer.update_nystrom_sketch(per_sample_grads)
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
        clip_threshold: float = 1.0,
        reg_lambda: float = 1e-4,
        landmark_selection: str = "topk_norm",
        min_dim_for_nystrom: int = 1024,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if nys_rank < 0:
            raise ValueError(f"Invalid Nyström rank: {nys_rank}")
        if landmark_selection not in ("topk_norm", "random", "leverage"):
            raise ValueError(f"Invalid landmark selection: {landmark_selection}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            nys_rank=nys_rank,
            nys_refresh_interval=nys_refresh_interval,
            nys_beta=nys_beta,
            clip_threshold=clip_threshold,
            reg_lambda=reg_lambda,
            landmark_selection=landmark_selection,
            min_dim_for_nystrom=min_dim_for_nystrom,
        )
        super().__init__(params, defaults)

        # Global step counter
        self._step_count = 0

        # Initialize Nyström state for each parameter group/parameter
        self._init_nystrom_state()

    def _init_nystrom_state(self) -> None:
        """Initialize Nyström sketch buffers for eligible parameters."""
        for group in self.param_groups:
            nys_rank = group["nys_rank"]
            min_dim = group["min_dim_for_nystrom"]

            for p in group["params"]:
                if p.requires_grad:
                    state = self.state[p]
                    d = p.numel()

                    # Determine if this parameter gets Nyström preconditioning
                    effective_rank = nys_rank if d >= min_dim else 0
                    state["nys_effective_rank"] = effective_rank

                    if effective_rank > 0:
                        # Q buffer: d × m sketch matrix (stored as flattened param × rank)
                        state["nys_Q"] = torch.zeros(
                            d, effective_rank, device=p.device, dtype=p.dtype
                        )
                        # Phi: m × m core matrix
                        state["nys_Phi"] = torch.eye(
                            effective_rank, device=p.device, dtype=p.dtype
                        ) * group["reg_lambda"]
                        # Flag for whether sketch has been initialized
                        state["nys_initialized"] = False

    @torch.no_grad()
    def step(self, closure=None):
        """
        Perform a single NysAdam optimization step.

        This method handles:
            1. Momentum update (first moment EMA)
            2. Second moment update (diagonal curvature proxy)
            3. Nyström-preconditioned update via Woodbury identity
            4. Element-wise clipping (Sophia-style trust region)
            5. Decoupled weight decay (AdamW-style)

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.

        Returns:
            loss: The loss value if closure is provided, else None.
        """
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
                if grad.is_sparse:
                    raise RuntimeError("NysAdam does not support sparse gradients.")

                state = self.state[p]

                # ---- State initialization (first call) ----
                if "step" not in state:
                    state["step"] = 0
                    # First moment (momentum)
                    state["exp_avg"] = torch.zeros_like(p)
                    # Second moment (diagonal curvature)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # ---- Step 2: Update first and second moments ----
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                m_hat = exp_avg / bias_correction1
                v_hat = exp_avg_sq / bias_correction2

                # ---- Step 4: Compute preconditioned update ----
                effective_rank = state.get("nys_effective_rank", 0)

                if effective_rank > 0 and state.get("nys_initialized", False):
                    # NysAdam update: low-rank + diagonal via Woodbury
                    update = self._woodbury_update(
                        m_hat=m_hat.view(-1),
                        v_hat=v_hat.view(-1),
                        Q=state["nys_Q"],
                        Phi=state["nys_Phi"],
                        eps=eps,
                    )
                    update = update.view_as(p)
                else:
                    # Pure Adam update (m=0 or sketch not yet initialized)
                    denom = v_hat.sqrt().add_(eps)
                    update = m_hat / denom

                # ---- Step 5: Element-wise clipping ----
                if clip_threshold < float("inf"):
                    update.clamp_(-clip_threshold, clip_threshold)

                # ---- Step 6: Apply update with decoupled weight decay ----
                if weight_decay > 0:
                    p.mul_(1 - lr * weight_decay)

                p.add_(update, alpha=-lr)

        return loss

    def _woodbury_update(
        self,
        m_hat: torch.Tensor,
        v_hat: torch.Tensor,
        Q: torch.Tensor,
        Phi: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """
        Compute N⁻¹ m̂ via the Sherman-Morrison-Woodbury identity.

        The preconditioner is:
            N = Q Φ⁻¹ Qᵀ + D²
        where D² = diag(v̂ + ε).

        By Woodbury:
            N⁻¹ x = D⁻² x - D⁻² Q (Φ + Qᵀ D⁻² Q)⁻¹ Qᵀ D⁻² x

        Args:
            m_hat: Bias-corrected first moment, shape [d].
            v_hat: Bias-corrected second moment, shape [d].
            Q: Nyström sketch matrix, shape [d, m].
            Phi: Core matrix, shape [m, m].
            eps: Stability constant.

        Returns:
            Preconditioned update vector, shape [d].
        """
        d, m = Q.shape

        # D⁻² = 1 / (v̂ + ε)  (element-wise)
        d_inv_sq = 1.0 / (v_hat + eps)  # [d]

        # z = D⁻² m̂   (the Adam step)
        z = d_inv_sq * m_hat  # [d]

        # Compute A = Φ + Qᵀ D⁻² Q  ∈ ℝ^{m×m}
        # First: D⁻¹ Q (scale each row of Q by d_inv_sq)
        D_inv_Q = Q * d_inv_sq.unsqueeze(1)  # [d, m]
        A = Phi + Q.t() @ D_inv_Q  # [m, m]

        # Solve A c = Qᵀ z  for c ∈ ℝ^m
        Qt_z = Q.t() @ z  # [m]

        # Use Cholesky for numerical stability (A is SPD)
        try:
            L = torch.linalg.cholesky(A)
            c = torch.cholesky_solve(Qt_z.unsqueeze(1), L).squeeze(1)  # [m]
        except torch.linalg.LinAlgError:
            # Fallback to general solve if Cholesky fails
            c = torch.linalg.solve(A, Qt_z)  # [m]

        # Woodbury correction: u = z - D⁻² Q c
        correction = D_inv_Q @ c  # [d]
        u = z - correction  # [d]

        return u

    @torch.no_grad()
    def update_nystrom_sketch(
        self,
        per_sample_grads: Dict[torch.nn.Parameter, torch.Tensor],
    ) -> None:
        """
        Update the Nyström sketch using per-sample gradients.

        This should be called every `nys_refresh_interval` steps.
        Per-sample gradients can be obtained via:
            - torch.func.vmap + torch.func.grad (PyTorch 2.0+)
            - Manual per-sample computation on a subsample

        Args:
            per_sample_grads: Dictionary mapping parameters to their
                per-sample gradients. Each tensor has shape [B, *param_shape]
                where B is the batch size.
        """
        for group in self.param_groups:
            nys_rank = group["nys_rank"]
            nys_beta = group["nys_beta"]
            reg_lambda = group["reg_lambda"]
            selection = group["landmark_selection"]

            for p in group["params"]:
                if p not in per_sample_grads:
                    continue

                state = self.state[p]
                effective_rank = state.get("nys_effective_rank", 0)

                if effective_rank == 0:
                    continue

                # Get per-sample gradients: [B, d]
                psg = per_sample_grads[p].view(per_sample_grads[p].size(0), -1)
                B, d = psg.shape
                m = effective_rank

                # Guard: skip if gradients contain NaN or Inf
                if not torch.isfinite(psg).all():
                    continue

                # ---- Step 3a: Select m landmark gradients ----
                # NOTE: if B < m, we can only get min(m, B) landmarks
                G_S = self._select_landmarks(psg, m, selection, state)  # [d, min(m,B)]
                actual_cols = G_S.shape[1]

                # ---- Step 3b: Update running sketch via EMA ----
                Q_old = state["nys_Q"]  # [d, m_actual]
                scale_new = math.sqrt(1 - nys_beta) / math.sqrt(B)

                if state.get("nys_initialized", False):
                    scale_old = math.sqrt(nys_beta)
                    # Concatenate: [d, Q_old_cols + actual_cols]
                    Q_combined = torch.cat(
                        [scale_old * Q_old, scale_new * G_S], dim=1
                    )
                else:
                    Q_combined = scale_new * G_S
                    state["nys_initialized"] = True

                # Truncated SVD to keep top-m singular vectors
                # Target rank is min(m, available columns) to avoid requesting
                # more components than exist
                target_rank = min(m, Q_combined.shape[1])
                Q_new = self._truncated_svd(Q_combined, target_rank)

                # Guard: if SVD produced NaN, keep the old sketch
                if torch.isfinite(Q_new).all():
                    state["nys_Q"] = Q_new
                else:
                    continue

                # ---- Step 3c: Compute core matrix ----
                # CRITICAL: use Q's ACTUAL column count, not the configured rank
                Q = state["nys_Q"]  # [d, actual_rank]
                actual_rank = Q.shape[1]
                state["nys_Phi"] = Q.t() @ Q + reg_lambda * torch.eye(
                    actual_rank, device=Q.device, dtype=Q.dtype
                )

    def _select_landmarks(
        self,
        psg: torch.Tensor,
        m: int,
        selection: str,
        state: dict,
    ) -> torch.Tensor:
        """
        Select m landmark gradients from per-sample gradients.

        Args:
            psg: Per-sample gradients, shape [B, d].
            m: Number of landmarks to select.
            selection: Strategy ('topk_norm', 'random', 'leverage').
            state: Optimizer state for this parameter.

        Returns:
            Selected landmarks, shape [d, m].
        """
        B, d = psg.shape
        m = min(m, B)  # Cannot select more than batch size

        if selection == "topk_norm":
            # Select gradients with largest norms (most informative)
            norms = psg.norm(dim=1)  # [B]
            _, indices = torch.topk(norms, m)
            G_S = psg[indices].t()  # [d, m]

        elif selection == "random":
            # Uniform random selection
            indices = torch.randperm(B, device=psg.device)[:m]
            G_S = psg[indices].t()  # [d, m]

        elif selection == "leverage":
            # Approximate leverage score sampling using current sketch
            Q = state.get("nys_Q", None)
            if Q is not None and state.get("nys_initialized", False):
                Phi = state["nys_Phi"]
                # Approximate leverage: ℓ_i = g_i^T (Q Φ⁻¹ Q^T + λI)⁻¹ g_i
                # Simplified: use ||Q^T g_i||² as proxy
                projections = Q.t() @ psg.t()  # [m, B]
                scores = projections.norm(dim=0)  # [B]
                # Sample proportional to scores
                probs = scores / scores.sum()
                indices = torch.multinomial(probs, m, replacement=False)
                G_S = psg[indices].t()
            else:
                # Fallback to random before first sketch
                indices = torch.randperm(B, device=psg.device)[:m]
                G_S = psg[indices].t()

        return G_S

    @staticmethod
    def _truncated_svd(M: torch.Tensor, rank: int) -> torch.Tensor:
        """
        Compute truncated SVD of M and return U * S^{1/2} for top-`rank`
        singular vectors. This gives the best rank-`rank` approximation
        such that Q Q^T ≈ M M^T.

        Args:
            M: Input matrix, shape [d, k] where k >= rank.
            rank: Target rank.

        Returns:
            Q: Shape [d, rank], satisfying Q Q^T ≈ M M^T (best rank approx).
        """
        d, k = M.shape
        rank = min(rank, k, d)

        try:
            # Use randomized SVD for efficiency when d >> rank
            if d > 4 * rank:
                U, S, _ = torch.svd_lowrank(M, q=rank)
            else:
                U, S, _ = torch.linalg.svd(M, full_matrices=False)
                U = U[:, :rank]
                S = S[:rank]
        except Exception:
            # Fallback: eigendecomposition of M M^T (small matrix)
            gram = M @ M.t()
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            # Take top-rank eigenvectors
            idx = torch.argsort(eigenvalues, descending=True)[:rank]
            U = eigenvectors[:, idx]
            S = eigenvalues[idx].clamp(min=0).sqrt()

        # Return U * sqrt(S) so that Q Q^T ≈ U S U^T
        Q = U * S.unsqueeze(0).sqrt()
        return Q

    @torch.no_grad()
    def compute_per_sample_grads_microbatch(
        self,
        model: torch.nn.Module,
        loss_fn,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        n_microbatches: int = 8,
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """
        Compute gradient directions via micro-batch splitting.

        Instead of B sequential single-sample backward passes (slow),
        split the batch into K micro-batches and compute one gradient
        per micro-batch. Each micro-batch gradient is an unbiased
        estimate of the full gradient, and their outer products capture
        the Fisher curvature.

        Complexity: K backward passes instead of B (typically 8 vs 32).
        Speedup: ~4x over single-sample sequential method.

        Args:
            model: The neural network model.
            loss_fn: Loss function.
            inputs: Input batch [B, ...].
            targets: Target batch [B, ...].
            n_microbatches: Number of gradient directions to compute.
                Each uses B/K samples. More = better sketch quality,
                fewer = faster (default: 8).

        Returns:
            Dictionary mapping parameters to micro-batch gradients
            of shape [n_microbatches, *param_shape].
        """
        B = inputs.size(0)
        K = min(n_microbatches, B)

        # Collect parameters eligible for Nystrom
        nys_params = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if state.get("nys_effective_rank", 0) > 0 and p.requires_grad:
                    nys_params.append(p)

        if not nys_params:
            return {}

        # Shuffle and split into K micro-batches
        perm = torch.randperm(B, device=inputs.device)
        chunk_size = B // K
        per_mb_grads = {p: [] for p in nys_params}

        for k in range(K):
            start = k * chunk_size
            end = start + chunk_size if k < K - 1 else B
            idx = perm[start:end]

            model.zero_grad()
            with torch.enable_grad():
                output = model(inputs[idx])
                loss_k = loss_fn(output, targets[idx])
                loss_k.backward()

            for p in nys_params:
                if p.grad is not None:
                    per_mb_grads[p].append(p.grad.detach().clone())

        result = {}
        for p in nys_params:
            if per_mb_grads[p]:
                result[p] = torch.stack(per_mb_grads[p], dim=0)

        return result

    # Keep the simple method as a fallback for small models / testing
    @torch.no_grad()
    def compute_per_sample_grads_simple(
        self,
        model: torch.nn.Module,
        loss_fn,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        subsample: Optional[int] = None,
    ) -> Dict[torch.nn.Parameter, torch.Tensor]:
        """
        Compute per-sample gradients via sequential backward passes.
        Use compute_per_sample_grads_microbatch for production (faster).

        Args:
            model: The neural network model.
            loss_fn: Loss function.
            inputs: Input batch [B, ...].
            targets: Target batch [B, ...].
            subsample: Compute for only this many samples.

        Returns:
            Dictionary mapping parameters to per-sample gradients [B, *shape].
        """
        B = inputs.size(0)

        if subsample is not None and subsample < B:
            indices = torch.randperm(B, device=inputs.device)[:subsample]
            inputs = inputs[indices]
            targets = targets[indices]
            B = subsample

        nys_params = []
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if state.get("nys_effective_rank", 0) > 0 and p.requires_grad:
                    nys_params.append(p)

        if not nys_params:
            return {}

        per_sample_grads = {p: [] for p in nys_params}

        for i in range(B):
            model.zero_grad()
            xi = inputs[i : i + 1]
            yi = targets[i : i + 1]
            with torch.enable_grad():
                output = model(xi)
                loss_i = loss_fn(output, yi)
                loss_i.backward()

            for p in nys_params:
                if p.grad is not None:
                    per_sample_grads[p].append(p.grad.detach().clone())

        result = {}
        for p in nys_params:
            if per_sample_grads[p]:
                result[p] = torch.stack(per_sample_grads[p], dim=0)

        return result

    def get_nystrom_diagnostics(self) -> Dict[str, float]:
        """
        Return diagnostic information about the Nystrom sketch quality.
        Safe against NaN values and cusolver errors.

        Returns:
            Dictionary with diagnostic metrics.
        """
        diagnostics = {
            "avg_sketch_rank": 0.0,
            "avg_condition_number": 0.0,
            "num_nystrom_layers": 0,
            "num_adam_layers": 0,
        }

        nys_count = 0
        total_cond = 0.0
        total_rank = 0.0

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                eff_rank = state.get("nys_effective_rank", 0)

                if eff_rank > 0 and state.get("nys_initialized", False):
                    diagnostics["num_nystrom_layers"] += 1
                    nys_count += 1

                    try:
                        Q = state["nys_Q"]
                        if not torch.isfinite(Q).all():
                            continue

                        # Use Phi eigenvalues instead of Q svdvals
                        # (Phi is m x m, much cheaper and stabler than Q which is d x m)
                        Phi = state["nys_Phi"]
                        eigvals = torch.linalg.eigvalsh(Phi)
                        pos_eigvals = eigvals[eigvals > 1e-10]
                        total_rank += len(pos_eigvals)

                        if len(pos_eigvals) > 0:
                            cond = (pos_eigvals[-1] / pos_eigvals[0]).item()
                            total_cond += cond
                    except Exception:
                        continue
                else:
                    diagnostics["num_adam_layers"] += 1

        if nys_count > 0:
            diagnostics["avg_sketch_rank"] = total_rank / nys_count
            diagnostics["avg_condition_number"] = total_cond / nys_count

        return diagnostics

    def __repr__(self) -> str:
        nys_rank = self.defaults["nys_rank"]
        lr = self.defaults["lr"]
        wd = self.defaults["weight_decay"]
        return (
            f"NysAdam(lr={lr}, nys_rank={nys_rank}, "
            f"weight_decay={wd}, "
            f"betas={self.defaults['betas']}, "
            f"clip={self.defaults['clip_threshold']})"
        )
