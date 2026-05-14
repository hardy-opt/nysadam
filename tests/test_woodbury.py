"""
Unit Tests: Woodbury Identity and Nyström Utilities
====================================================
Author: Hardik Tankaria
Date: May 2026
Description:
    Numerical correctness tests for core linear algebra operations.

Usage: pytest tests/test_woodbury.py -v
"""

import pytest
import torch
import math


class TestWoodburySolve:
    """Test the standalone Woodbury solver."""

    def test_matches_direct_solve(self):
        from nysadam.utils.nystrom_utils import woodbury_solve

        torch.manual_seed(42)
        d, m = 50, 5

        Q = torch.randn(d, m) * 0.5
        Phi = Q.t() @ Q + 0.01 * torch.eye(m)
        D_diag = torch.rand(d) + 0.1
        rhs = torch.randn(d)

        # Woodbury
        x_wb = woodbury_solve(Q, Phi, D_diag, rhs)

        # Direct
        D = torch.diag(D_diag)
        Phi_inv = torch.linalg.inv(Phi)
        N = Q @ Phi_inv @ Q.t() + D
        x_direct = torch.linalg.solve(N, rhs)

        torch.testing.assert_close(x_wb, x_direct, atol=1e-4, rtol=1e-3)

    def test_identity_preconditioner(self):
        """With Q=0, should reduce to D^{-1} rhs."""
        from nysadam.utils.nystrom_utils import woodbury_solve

        d, m = 30, 3
        Q = torch.zeros(d, m)
        Phi = torch.eye(m)
        D_diag = torch.rand(d) + 0.5
        rhs = torch.randn(d)

        x = woodbury_solve(Q, Phi, D_diag, rhs)
        expected = rhs / D_diag
        torch.testing.assert_close(x, expected, atol=1e-6, rtol=1e-5)

    def test_symmetry(self):
        """N^{-1} should be symmetric when N is symmetric."""
        from nysadam.utils.nystrom_utils import woodbury_solve

        torch.manual_seed(42)
        d, m = 20, 3

        Q = torch.randn(d, m)
        Phi = Q.t() @ Q + 0.1 * torch.eye(m)
        D_diag = torch.rand(d) + 0.1

        # Compute N^{-1} column by column
        N_inv = torch.zeros(d, d)
        for i in range(d):
            e_i = torch.zeros(d)
            e_i[i] = 1.0
            N_inv[:, i] = woodbury_solve(Q, Phi, D_diag, e_i)

        # Check symmetry
        asym = (N_inv - N_inv.t()).abs().max().item()
        assert asym < 1e-4, f"N^{{-1}} not symmetric: max asymmetry = {asym}"


class TestEffectiveDimension:
    """Test effective dimension computation."""

    def test_identity_hessian(self):
        """d_eff of identity = d / (1 + λ)."""
        from nysadam.utils.nystrom_utils import compute_effective_dimension

        d = 100
        eigenvalues = torch.ones(d)
        lam = 1.0
        d_eff = compute_effective_dimension(eigenvalues, lam)
        expected = d * 1.0 / (1.0 + lam)  # 100 * 0.5 = 50
        assert abs(d_eff - expected) < 1e-4

    def test_decaying_spectrum(self):
        """d_eff should be much less than d for rapidly decaying spectrum."""
        from nysadam.utils.nystrom_utils import compute_effective_dimension

        d = 1000
        eigenvalues = torch.tensor([1.0 / (i + 1) ** 2 for i in range(d)])
        lam = 0.01
        d_eff = compute_effective_dimension(eigenvalues, lam)

        # With rapid decay, d_eff << d
        assert d_eff < d * 0.1, f"d_eff = {d_eff}, expected much less than {d}"

    def test_monotone_in_lambda(self):
        """d_eff should decrease as λ increases."""
        from nysadam.utils.nystrom_utils import compute_effective_dimension

        eigenvalues = torch.rand(100) + 0.01
        d1 = compute_effective_dimension(eigenvalues, 0.01)
        d2 = compute_effective_dimension(eigenvalues, 0.1)
        d3 = compute_effective_dimension(eigenvalues, 1.0)
        assert d1 > d2 > d3


class TestTruncatedSVD:
    """Test truncated SVD utility."""

    def test_rank_constraint(self):
        from nysadam.utils.nystrom_utils import truncated_svd_stable

        M = torch.randn(100, 20)
        U, S = truncated_svd_stable(M, rank=5)
        assert U.shape == (100, 5)
        assert S.shape == (5,)

    def test_descending_singular_values(self):
        from nysadam.utils.nystrom_utils import truncated_svd_stable

        M = torch.randn(50, 30)
        _, S = truncated_svd_stable(M, rank=10)
        for i in range(len(S) - 1):
            assert S[i] >= S[i + 1] - 1e-6

    def test_orthonormal_left_vectors(self):
        from nysadam.utils.nystrom_utils import truncated_svd_stable

        M = torch.randn(100, 20)
        U, _ = truncated_svd_stable(M, rank=5)
        # U^T U should be close to identity
        gram = U.t() @ U
        identity = torch.eye(5)
        diff = (gram - identity).abs().max().item()
        assert diff < 1e-4, f"U not orthonormal: max error = {diff}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
