"""
Reproducibility and Device Utilities
=====================================
Author: Hardik Tankaria | Date: May 2026
Description:
    Deterministic seeding, device selection, and multi-GPU setup.
"""

import os
import random
import logging
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("nysadam")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all random seeds for reproducibility across devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_device() -> torch.device:
    """Get the primary computation device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_gpu_count() -> int:
    """Return the number of available CUDA GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def setup_model_for_device(
    model: nn.Module,
    device: torch.device,
) -> Tuple[nn.Module, nn.Module]:
    """
    Move model to device and wrap with DataParallel when multiple
    GPUs are available.

    DataParallel splits each batch across GPUs along dimension 0,
    runs forward on each shard in parallel, then gathers and reduces
    the loss on the primary GPU. Backward is handled automatically.

    Args:
        model: Neural network.
        device: Primary device.

    Returns:
        (wrapped_model, raw_model) tuple.
        - wrapped_model: DataParallel wrapper (use for forward pass)
        - raw_model: unwrapped module (use for optimizer param groups
          and per-sample gradient computation)

    The distinction matters because:
        optimizer = NysAdam(raw_model.parameters(), ...)
        loss = loss_fn(wrapped_model(inputs), targets)
    DataParallel wraps parameters in a .module attribute. The optimizer
    must reference the original parameters, not the wrapper.
    """
    model = model.to(device)
    raw_model = model

    n_gpus = get_gpu_count()
    if n_gpus > 1 and device.type == "cuda":
        model = nn.DataParallel(model)
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        logger.info(f"DataParallel enabled across {n_gpus} GPUs: {gpu_names}")
        print(f"  Multi-GPU: DataParallel across {n_gpus} GPUs "
              f"({', '.join(gpu_names)})")

    return model, raw_model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_gpu_summary() -> None:
    """Print GPU information for experiment logging."""
    if not torch.cuda.is_available():
        print("  Device: CPU")
        return

    n_gpus = torch.cuda.device_count()
    for i in range(n_gpus):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
        print(f"  GPU {i}: {name} ({mem:.1f} GB)")
