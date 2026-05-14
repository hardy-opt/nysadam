"""
NysAdam: Nyström-Augmented Adaptive Moment Estimation
=====================================================
Author: Hardik Tankaria
Date: May 2026
Description: A regularized Nyström-preconditioned optimizer for non-convex
             optimization in ML, DL, and LLMs.

This package implements NysAdam and provides experiment infrastructure for
benchmarking against AdamW, Sophia, Shampoo, and SketchySGD on classical ML,
deep learning (CIFAR, ImageNet), PINNs, and LLM fine-tuning tasks.
"""

from nysadam.optimizers.nysadam import NysAdam
from nysadam.optimizers.nysadam_variants import NysAdamLight, NysSophia

__version__ = "0.1.0"
__author__ = "Hardik Tankaria"

__all__ = ["NysAdam", "NysAdamLight", "NysSophia"]
