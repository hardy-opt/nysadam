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
    """
    Compute total gradient L2 norm across all parameters.

    Uses a single fused norm over all gradients on each device to avoid
    issuing one host sync per parameter. For ResNet-18 (~60 leaf params)
    this collapses 60 syncs into 1 per call.
    """
    grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
    if not grads:
        return 0.0
    # Stack on the device the grads live on. torch.norm of a vector
    # of norms equals the L2 norm of the concatenation.
    device = grads[0].device
    norms = torch.stack(
        [torch.linalg.vector_norm(g, ord=2) for g in grads]
    )
    return torch.linalg.vector_norm(norms, ord=2).item()


def compute_param_norm(model: torch.nn.Module) -> float:
    """Compute total parameter L2 norm."""
    params = [p.detach() for p in model.parameters()]
    if not params:
        return 0.0
    norms = torch.stack([torch.linalg.vector_norm(p, ord=2) for p in params])
    return torch.linalg.vector_norm(norms, ord=2).item()


def get_peak_memory_mb() -> float:
    """Get peak GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


@torch.inference_mode()
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

    Memory note: F1 and ECE are computed on streaming confusion-matrix /
    bin counts rather than by concatenating all per-batch predictions.
    For CIFAR-100 over 10k samples this saves ~4 MB peak per eval; for
    LM vocabularies it saves orders of magnitude more.

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
    is_lm: Optional[bool] = None  # Auto-detect on first batch

    # Streaming classification accumulators (constant memory in N samples).
    # Allocated lazily once num_classes is known.
    confusion: Optional[torch.Tensor] = None  # [C, C], rows=true, cols=pred
    ece_bins = 15
    ece_bin_conf_sum: Optional[torch.Tensor] = None  # [bins]
    ece_bin_correct_sum: Optional[torch.Tensor] = None  # [bins]
    ece_bin_count: Optional[torch.Tensor] = None  # [bins]
    ece_boundaries: Optional[torch.Tensor] = None

    for batch in dataloader:
        if len(batch) == 2:
            inputs, targets = batch
        else:
            inputs, targets = batch[0], batch[1]

        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Auto-detect task type on first batch
        if is_lm is None:
            is_lm = (outputs.dim() == 3 and outputs.size(-1) > 1000)

        if is_lm:
            # Language model: streaming token accuracy
            token_preds = outputs.argmax(dim=-1)  # [batch, seq_len]
            total_tokens += targets.numel()
            total_correct_tokens += (token_preds == targets).sum().item()
        elif outputs.dim() > 1 and outputs.size(1) > 1:
            # Classification: streaming confusion matrix + ECE bin sums.
            # No per-batch tensors are retained; each batch contributes
            # to running totals and is then released.
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            C = probs.size(1) if num_classes <= 0 else num_classes
            if confusion is None:
                confusion = torch.zeros(C, C, dtype=torch.long, device=device)
                ece_bin_conf_sum = torch.zeros(ece_bins, device=device)
                ece_bin_correct_sum = torch.zeros(ece_bins, device=device)
                ece_bin_count = torch.zeros(ece_bins, dtype=torch.long, device=device)
                ece_boundaries = torch.linspace(0, 1, ece_bins + 1, device=device)

            # Accumulate confusion matrix via flat-index scatter_add.
            flat = targets * C + preds
            confusion.view(-1).scatter_add_(
                0, flat, torch.ones_like(flat, dtype=torch.long)
            )

            # ECE bins: index into [0, ece_bins-1] from confidence
            bin_idx = torch.bucketize(confidences, ece_boundaries[1:-1])
            correct = (preds == targets).float()
            ece_bin_conf_sum.scatter_add_(0, bin_idx, confidences)
            ece_bin_correct_sum.scatter_add_(0, bin_idx, correct)
            ece_bin_count.scatter_add_(
                0, bin_idx, torch.ones_like(bin_idx, dtype=torch.long)
            )

        # Drop per-batch tensors before next iteration so the autograd
        # graph and large logit tensors are reclaimed promptly.
        del outputs, loss

    model.train()

    result: Dict[str, float] = {"loss": total_loss / max(total_samples, 1)}

    if is_lm:
        import math
        result["perplexity"] = math.exp(min(result["loss"], 20))
        result["token_accuracy"] = total_correct_tokens / max(total_tokens, 1)
    elif confusion is not None:
        confusion_cpu = confusion.cpu()
        C = confusion_cpu.size(0)
        n_total = int(confusion_cpu.sum().item())

        correct = int(confusion_cpu.diag().sum().item())
        result["accuracy"] = correct / max(n_total, 1)
        result["f1_macro"] = _f1_macro_from_confusion(confusion_cpu)

        per_class_acc: Dict[int, float] = {}
        row_totals = confusion_cpu.sum(dim=1)
        for c in range(C):
            if row_totals[c].item() > 0:
                per_class_acc[c] = (
                    confusion_cpu[c, c].item() / row_totals[c].item()
                )
        result["per_class_accuracy"] = per_class_acc

        result["ece"] = _ece_from_bins(
            ece_bin_conf_sum.cpu(),
            ece_bin_correct_sum.cpu(),
            ece_bin_count.cpu(),
        )

    return result


def _f1_macro_from_confusion(confusion: torch.Tensor) -> float:
    """
    Compute macro-averaged F1 from a confusion matrix.

    F1-Macro averages F1 across all classes equally, making it robust
    to class imbalance unlike accuracy which is dominated by the
    majority class.

    Args:
        confusion: [C, C] confusion matrix (rows=true, cols=predicted).

    Returns:
        Macro F1 averaged over classes that appear in the targets.
    """
    eps = 1e-8
    tp = confusion.diag().float()
    fp = confusion.sum(dim=0).float() - tp
    fn = confusion.sum(dim=1).float() - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    # Only average over classes that exist in the targets
    present = confusion.sum(dim=1) > 0
    if not present.any():
        return 0.0
    return f1[present].mean().item()


def _ece_from_bins(
    conf_sum: torch.Tensor,
    correct_sum: torch.Tensor,
    counts: torch.Tensor,
) -> float:
    """
    Compute Expected Calibration Error from streaming bin accumulators.

    ECE measures how well predicted probabilities match actual
    correctness frequencies. A well-calibrated model has ECE ≈ 0.

    Args:
        conf_sum: Sum of confidences per bin [n_bins].
        correct_sum: Sum of correct predictions per bin [n_bins].
        counts: Number of samples per bin [n_bins].

    Returns:
        ECE value (lower is better, 0 = perfectly calibrated).
    """
    total = counts.sum().item()
    if total == 0:
        return 0.0

    ece = 0.0
    for i in range(counts.numel()):
        n = counts[i].item()
        if n > 0:
            avg_conf = conf_sum[i].item() / n
            avg_acc = correct_sum[i].item() / n
            ece += (n / total) * abs(avg_acc - avg_conf)
    return ece
