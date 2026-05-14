"""
Metrics and Logging Utilities
=============================
Author: Hardik Tankaria
Date: May 2026
Description:
    Training metrics collection, logging, and visualization for
    optimizer comparison experiments.
"""

import time
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

import torch
import numpy as np


@dataclass
class StepMetrics:
    """Metrics for a single training step."""
    step: int
    epoch: int
    train_loss: float
    learning_rate: float
    grad_norm: float
    update_norm: float
    wall_time: float
    # Optional
    train_accuracy: Optional[float] = None
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    # NysAdam-specific
    nys_condition_number: Optional[float] = None
    nys_effective_rank: Optional[float] = None
    peak_memory_mb: Optional[float] = None


@dataclass
class ExperimentLog:
    """Complete log for one experiment run."""
    optimizer_name: str
    dataset: str
    model: str
    config: Dict
    steps: List[StepMetrics] = field(default_factory=list)
    total_time: float = 0.0
    best_val_loss: float = float("inf")
    best_val_accuracy: float = 0.0
    final_train_loss: float = float("inf")
    final_test_loss: float = float("inf")
    final_test_accuracy: float = 0.0

    def add_step(self, metrics: StepMetrics) -> None:
        self.steps.append(metrics)
        if metrics.val_loss is not None:
            self.best_val_loss = min(self.best_val_loss, metrics.val_loss)
        if metrics.val_accuracy is not None:
            self.best_val_accuracy = max(self.best_val_accuracy, metrics.val_accuracy)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "optimizer_name": self.optimizer_name,
            "dataset": self.dataset,
            "model": self.model,
            "config": self.config,
            "total_time": self.total_time,
            "best_val_loss": self.best_val_loss,
            "best_val_accuracy": self.best_val_accuracy,
            "final_train_loss": self.final_train_loss,
            "final_test_loss": self.final_test_loss,
            "final_test_accuracy": self.final_test_accuracy,
            "steps": [asdict(s) for s in self.steps],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "ExperimentLog":
        with open(path) as f:
            data = json.load(f)
        log = cls(
            optimizer_name=data["optimizer_name"],
            dataset=data["dataset"],
            model=data["model"],
            config=data["config"],
            total_time=data.get("total_time", 0.0),
            best_val_loss=data.get("best_val_loss", float("inf")),
            best_val_accuracy=data.get("best_val_accuracy", 0.0),
            final_train_loss=data.get("final_train_loss", float("inf")),
            final_test_loss=data.get("final_test_loss", float("inf")),
            final_test_accuracy=data.get("final_test_accuracy", 0.0),
        )
        for s in data.get("steps", []):
            log.steps.append(StepMetrics(**s))
        return log


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self):
        self.elapsed = 0.0
        self._start = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self._start


def compute_grad_norm(model: torch.nn.Module) -> float:
    """Compute total gradient L2 norm across all parameters."""
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def compute_param_norm(model: torch.nn.Module) -> float:
    """Compute total parameter L2 norm."""
    total = 0.0
    for p in model.parameters():
        total += p.data.norm(2).item() ** 2
    return total ** 0.5


def get_peak_memory_mb() -> float:
    """Get peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    loss_fn,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Returns:
        Dictionary with 'loss' and 'accuracy' (if classification).
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in dataloader:
        if len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch[0], batch[1]

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        # Classification accuracy
        if outputs.dim() > 1 and outputs.size(1) > 1:
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()

    model.train()

    result = {"loss": total_loss / max(total_samples, 1)}
    if total_correct > 0 or (total_samples > 0 and outputs.dim() > 1):
        result["accuracy"] = total_correct / max(total_samples, 1)

    return result
