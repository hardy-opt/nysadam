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
    num_classes: int = 0,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset with comprehensive metrics.

    For classification (outputs are [batch, num_classes]):
    - Loss, Accuracy, F1-Macro, ECE, per-class accuracy

    For language modeling (outputs are [batch, seq_len, vocab_size]):
    - Loss, Perplexity, token-level accuracy
    - Skips ECE/F1/per-class (not applicable, and storing [N, 50K]
      probability vectors would exhaust CPU memory)

    Following Gneiting & Raftery (2007): loss is the primary metric
    for model selection since cross-entropy is a strictly proper
    scoring rule.

    Args:
        model: Neural network.
        dataloader: Evaluation data loader.
        loss_fn: Loss function.
        device: Computation device.
        num_classes: Number of classes (auto-detected if 0).

    Returns:
        Dictionary with metrics appropriate to the task.
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_tokens = 0
    total_correct_tokens = 0
    is_lm = None  # Auto-detect on first batch

    # Classification accumulators (only used for non-LM tasks)
    all_preds = []
    all_targets = []
    all_probs = []

    for batch in dataloader:
        if len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch[0], batch[1]

        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Auto-detect task type on first batch
        if is_lm is None:
            is_lm = (outputs.dim() == 3 and outputs.size(-1) > 1000)

        if is_lm:
            # Language model: compute token accuracy without storing probs
            # outputs: [batch, seq_len, vocab_size]
            # targets: [batch, seq_len]
            token_preds = outputs.argmax(dim=-1)  # [batch, seq_len]
            total_tokens += targets.numel()
            total_correct_tokens += (token_preds == targets).sum().item()
        else:
            # Classification: accumulate for F1/ECE (small tensors)
            if outputs.dim() > 1 and outputs.size(1) > 1:
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(targets.cpu())
                all_probs.append(probs.cpu())

    model.train()

    result = {"loss": total_loss / max(total_samples, 1)}

    if is_lm:
        # Language model metrics
        import math
        result["perplexity"] = math.exp(min(result["loss"], 20))  # Cap to avoid overflow
        result["token_accuracy"] = total_correct_tokens / max(total_tokens, 1)
    elif all_preds:
        # Classification metrics
        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)
        probs = torch.cat(all_probs)

        correct = (preds == targets).sum().item()
        result["accuracy"] = correct / len(targets)

        n_classes = num_classes if num_classes > 0 else probs.size(1)
        result["f1_macro"] = _compute_f1_macro(preds, targets, n_classes)

        per_class_acc = {}
        for c in range(n_classes):
            mask = targets == c
            if mask.sum() > 0:
                per_class_acc[c] = (preds[mask] == c).float().mean().item()
        result["per_class_accuracy"] = per_class_acc

        result["ece"] = _compute_ece(probs, targets, n_bins=15)

    return result


def _compute_f1_macro(
    preds: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
) -> float:
    """
    Compute macro-averaged F1 score.

    F1-Macro averages F1 across all classes equally, making it
    robust to class imbalance — unlike accuracy which is dominated
    by the majority class.
    """
    f1_scores = []
    for c in range(n_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        # Only include classes that exist in the targets
        if (targets == c).sum() > 0:
            f1_scores.append(f1.item())

    return sum(f1_scores) / max(len(f1_scores), 1)


def _compute_ece(
    probs: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (Guo et al. 2017).

    ECE measures how well predicted probabilities match actual
    correctness frequencies. A well-calibrated model has ECE ≈ 0.

    A model selected at minimum validation loss tends to have lower ECE
    than one selected at maximum accuracy, because loss penalizes
    overconfident miscalibrated predictions while accuracy does not.

    Args:
        probs: Predicted probabilities [N, C].
        targets: True labels [N].
        n_bins: Number of confidence bins.

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated).
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets).float()

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        n_in_bin = in_bin.sum().item()
        if n_in_bin > 0:
            avg_confidence = confidences[in_bin].mean().item()
            avg_accuracy = accuracies[in_bin].mean().item()
            ece += (n_in_bin / len(targets)) * abs(avg_accuracy - avg_confidence)

    return ece
