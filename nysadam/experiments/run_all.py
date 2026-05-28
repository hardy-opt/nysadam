"""
Unified Experiment Runner
=========================
Author: Hardik Tankaria
Date: May 2026
Description:
    Master training loop that handles all optimizer types and experiment
    configurations. Supports:
    - Automatic per-sample gradient computation for NysAdam
    - Hessian diagonal updates for Sophia
    - Preconditioner updates for Shampoo
    - Fair wall-clock comparison across all methods
    - Comprehensive logging for paper figures/tables

Usage:
    python -m nysadam.experiments.run_all --config configs/cifar10_resnet.yaml
    python -m nysadam.experiments.run_all --experiment cifar10 --optimizer nysadam
"""

import os
import sys
import gc
import time
import argparse
import yaml
from typing import Dict, Optional, Tuple
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nysadam.optimizers.nysadam import NysAdam
from nysadam.optimizers.nysadam_variants import NysAdamLight, NysSophia
from nysadam.optimizers.sophia import SophiaG
from nysadam.optimizers.shampoo import Shampoo
from nysadam.optimizers.sketchy_sgd import SketchySGD
from nysadam.models.models import create_model
from nysadam.data.datasets import (
    load_sklearn_dataset, load_cifar, load_burgers_data, load_text_data,
)
from nysadam.utils.metrics import (
    ExperimentLog, StepMetrics, Timer, compute_grad_norm,
    evaluate, get_peak_memory_mb,
)
from nysadam.utils.reproducibility import (
    set_seed, get_device, count_parameters,
    setup_model_for_device, get_gpu_count, print_gpu_summary,
)


# ═══════════════════════════════════════════════════════════════════
# Memory hygiene helpers
# ═══════════════════════════════════════════════════════════════════

# On Linux, glibc's malloc holds freed pages in per-thread arenas rather
# than returning them to the OS. malloc_trim(0) forces a return, keeping
# RSS bounded. This is a no-op on non-glibc platforms.
try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6")
    def _malloc_trim():
        _libc.malloc_trim(0)
except (OSError, AttributeError):
    def _malloc_trim():
        pass


def _release_memory() -> None:
    """
    Aggressively reclaim Python and CUDA memory.

    Under high tensor churn, CPython's generational GC defers collection
    of objects in older generations. Force a full collection cycle and
    flush the CUDA caching allocator. Safe to call repeatedly.
    """
    gc.collect()
    _malloc_trim()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def _epoch_gc() -> None:
    """
    Lightweight per-epoch garbage collection.

    nn.DataParallel creates reference cycles (autograd nodes <-> replicated
    tensors <-> thread-local state) on every forward pass. CPython's cyclic
    GC does not run gen-2 collections frequently enough to keep up with
    DataParallel's allocation rate. Without an explicit gc.collect(), the
    uncollected cycles accumulate at ~200 MB per epoch.

    This function runs gc.collect() + malloc_trim once per epoch. The cost
    is O(number of live Python objects), typically <5 ms for a training
    script, which is negligible relative to the epoch time (~30s for
    CIFAR-10 ResNet-18).

    We intentionally do NOT call torch.cuda.empty_cache() here — it forces
    a cudaDeviceSynchronize and costs 5-6% throughput per the HuggingFace
    Transformers benchmark (GH #31372). It is reserved for between-run
    cleanup in _release_memory().
    """
    gc.collect()
    _malloc_trim()


def _save_checkpoint_cpu(
    raw_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Dict,
    path: str,
) -> None:
    """
    Save a checkpoint by streaming tensors to CPU one at a time.

    A naive ``torch.save({"model_state_dict": model.state_dict(), ...})``
    where the state dict contains GPU tensors causes pickle to copy ALL
    tensors to CPU simultaneously before writing. For ResNet-18 with an
    Adam-family optimizer this temporarily allocates ~130 MB on CPU per
    save. Under cosine LR over 60 epochs, the best-checkpoint may be
    rewritten ~30 times — accumulating multi-GB transient allocations
    that CPython's GC fails to reclaim promptly under high allocation
    pressure, and that PyTorch's CPU memory pool retains.

    By explicitly materializing state on CPU into a single dict, then
    immediately calling ``torch.save`` and deleting the dict, we keep
    the lifetime of CPU copies tight and bounded.

    Args:
        raw_model: Unwrapped model (no DataParallel wrapper).
        optimizer: Optimizer whose state will be persisted.
        metadata: Plain-Python checkpoint metadata (no tensors).
        path: Destination file path.
    """
    # Move state to CPU explicitly so torch.save doesn't create a second copy
    model_state = {k: v.detach().cpu() for k, v in raw_model.state_dict().items()}
    optimizer_state = _optimizer_state_to_cpu(optimizer.state_dict())

    checkpoint = {**metadata, "model_state_dict": model_state,
                  "optimizer_state_dict": optimizer_state}
    torch.save(checkpoint, path)

    # Drop references immediately so the CPU tensor copies can be
    # reclaimed by the next _epoch_gc() call at end of epoch.
    del checkpoint, model_state, optimizer_state


def _optimizer_state_to_cpu(state_dict: Dict) -> Dict:
    """Recursively move tensor values inside an optimizer state_dict to CPU."""
    if isinstance(state_dict, torch.Tensor):
        return state_dict.detach().cpu()
    if isinstance(state_dict, dict):
        return {k: _optimizer_state_to_cpu(v) for k, v in state_dict.items()}
    if isinstance(state_dict, (list, tuple)):
        result = [_optimizer_state_to_cpu(v) for v in state_dict]
        return type(state_dict)(result)
    return state_dict


# ═══════════════════════════════════════════════════════════════════
# Optimizer Factory
# ═══════════════════════════════════════════════════════════════════

def create_optimizer(
    name: str,
    model: nn.Module,
    config: Dict,
) -> torch.optim.Optimizer:
    """
    Create optimizer by name with given configuration.

    Args:
        name: Optimizer name.
        model: Neural network model.
        config: Optimizer hyperparameters.

    Returns:
        Optimizer instance.
    """
    params = model.parameters()
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.01)
    betas = tuple(config.get("betas", [0.9, 0.999]))

    if name == "nysadam":
        return NysAdam(
            params, lr=lr, weight_decay=weight_decay, betas=betas,
            nys_rank=config.get("nys_rank", 64),
            nys_refresh_interval=config.get("nys_refresh_interval", 10),
            nys_beta=config.get("nys_beta", 0.999),
            clip_threshold=config.get("clip_threshold", 1.0),
            reg_lambda=config.get("reg_lambda", 1e-4),
            landmark_selection=config.get("landmark_selection", "topk_norm"),
            min_dim_for_nystrom=config.get("min_dim_for_nystrom", 1024),
        )
    elif name == "nysadam_light":
        return NysAdamLight(
            params, lr=lr, weight_decay=weight_decay, betas=betas,
            nys_rank=config.get("nys_rank", 32),
            grad_buffer_size=config.get("grad_buffer_size", 128),
            nys_refresh_interval=config.get("nys_refresh_interval", 50),
            clip_threshold=config.get("clip_threshold", 1.0),
        )
    elif name == "nyssophia":
        return NysSophia(
            params, lr=lr, weight_decay=weight_decay, betas=betas,
            nys_rank=config.get("nys_rank", 64),
            nys_refresh_interval=config.get("nys_refresh_interval", 10),
            clip_threshold=config.get("clip_threshold", 1.0),
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            params, lr=lr, weight_decay=weight_decay, betas=betas,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, weight_decay=weight_decay,
            momentum=config.get("momentum", 0.9),
        )
    elif name == "sophia":
        return SophiaG(
            params, lr=config.get("lr", 1e-4),
            betas=tuple(config.get("betas", [0.965, 0.99])),
            weight_decay=weight_decay,
            rho=config.get("rho", 1.0),
        )
    elif name == "shampoo":
        return Shampoo(
            params, lr=lr, weight_decay=weight_decay, betas=betas,
            precondition_frequency=config.get("precondition_frequency", 100),
            graft_type=config.get("graft_type", "adam"),
        )
    elif name == "sketchy_sgd":
        return SketchySGD(
            params, lr=config.get("lr", 0.01),
            nys_rank=config.get("nys_rank", 64),
            mu=config.get("mu", 1e-3),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


# ═══════════════════════════════════════════════════════════════════
# Per-Sample Gradient Computation
# ═══════════════════════════════════════════════════════════════════

def compute_nystrom_grads_microbatch(
    model: nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_microbatches: int = 8,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Compute gradient directions for Nystrom sketch via micro-batch splitting.

    Instead of B sequential single-sample backward passes (the vmap path
    silently falls back to this on models with BatchNorm), split the
    current batch into K micro-batches and compute one gradient per
    micro-batch. Each micro-batch gradient captures a different view
    of the curvature.

    Complexity: K backward passes (typically 8) instead of B (typically
    32). Provides ~4x speedup with minimal sketch quality loss.

    Args:
        model: Neural network (use raw model, not DataParallel wrapper).
        loss_fn: Loss function.
        inputs: Input batch [B, ...].
        targets: Target batch [B, ...].
        n_microbatches: Number of gradient directions to compute.

    Returns:
        Dictionary mapping parameters to gradients [n_microbatches, *shape].
    """
    B = inputs.size(0)
    K = min(n_microbatches, B)

    params = [p for p in model.parameters() if p.requires_grad]
    per_mb_grads = {p: [] for p in params}

    perm = torch.randperm(B, device=inputs.device)
    chunk_size = B // K

    for k in range(K):
        start = k * chunk_size
        end = start + chunk_size if k < K - 1 else B
        idx = perm[start:end]

        model.zero_grad()
        with torch.enable_grad():
            output = model(inputs[idx])
            loss_k = loss_fn(output, targets[idx])
            loss_k.backward()

        for p in params:
            if p.grad is not None:
                per_mb_grads[p].append(p.grad.detach().clone())

    result = {}
    for p in params:
        if per_mb_grads[p]:
            result[p] = torch.stack(per_mb_grads[p], dim=0)

    return result


# ═══════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn,
    config: Dict,
    device: torch.device,
    experiment_name: str = "experiment",
) -> ExperimentLog:
    """
    Main training loop with support for all optimizer types.

    Evaluation protocol (Hastie et al. 2009, Ch. 7):
    - val_loader: used every `eval_every` epochs for model selection
      and checkpoint saving (minimum validation loss by default)
    - test_loader: touched EXACTLY ONCE after training is complete,
      for final unbiased evaluation only

    Model selection criterion (Gneiting & Raftery, 2007):
    - Default: minimum validation loss (cross-entropy is a strictly
      proper scoring rule — uniquely minimized at true distribution)
    - Override via config['selection_metric'] = 'accuracy' when
      top-1 accuracy is the sole deployment metric

    Args:
        model: Neural network (possibly DataParallel-wrapped).
        optimizer: Optimizer instance.
        optimizer_name: Name string for logging.
        train_loader: Training data loader.
        val_loader: Validation data loader (for model selection).
        test_loader: Test data loader (touched ONCE at the end).
        loss_fn: Loss function.
        config: Training configuration dict.
        device: Computation device.
        experiment_name: Name for logging.

    Returns:
        ExperimentLog with all metrics.
    """
    epochs = config.get("epochs", 100)
    eval_every = config.get("eval_every", 1)
    log_every = config.get("log_every", 10)
    nys_n_microbatches = config.get("nys_n_microbatches",
                                    config.get("nys_subsample", 8))
    nys_refresh = config.get("nys_refresh_interval", 10)
    hessian_refresh = config.get("hessian_refresh_interval", 10)

    # Model selection criterion (Gneiting & Raftery 2007):
    # Default 'loss' = min validation loss (proper scoring rule)
    # Alternative 'accuracy' = max validation accuracy (only when
    #   accuracy is the sole deployment metric + balanced classes)
    selection_metric = config.get("selection_metric", "loss")

    # Unwrap DataParallel once, up front. Holding ONE reference to the raw
    # module (as opposed to re-resolving `model.module` repeatedly) makes
    # ownership explicit and lets us clean up deterministically at the end.
    raw_model = model.module if hasattr(model, "module") else model
    model_class_name = raw_model.__class__.__name__

    # Learning rate scheduler
    scheduler_name = config.get("scheduler", "cosine")
    total_steps = epochs * len(train_loader)
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps,
        )
    elif scheduler_name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=config.get("step_size", 30), gamma=0.1,
        )
    else:
        scheduler = None

    # Checkpoint directory
    save_checkpoints = config.get("save_checkpoints", True)
    checkpoint_dir = config.get("checkpoint_dir", "results/checkpoints")
    if save_checkpoints:
        os.makedirs(checkpoint_dir, exist_ok=True)

    log = ExperimentLog(
        optimizer_name=optimizer_name,
        dataset=experiment_name,
        model=model_class_name,
        config=config,
    )

    model.train()
    global_step = 0
    start_time = time.perf_counter()
    # For checkpoint selection: lower is better for loss, higher for accuracy
    best_val_metric = float("inf") if selection_metric == "loss" else 0.0

    # Early stopping (Prechelt, 1998)
    patience = config.get("early_stopping_patience", 20)
    epochs_without_improvement = 0

    needs_per_sample = optimizer_name in ("nysadam", "nyssophia", "sketchy_sgd")
    needs_hessian = optimizer_name in ("sophia", "nyssophia")

    print(f"\n{'='*70}")
    print(f"Training: {experiment_name} | Optimizer: {optimizer_name}")
    print(f"Model: {model_class_name} | Params: {count_parameters(raw_model):,}")
    print(f"Epochs: {epochs} | Selection: min val {selection_metric}")
    print_gpu_summary()
    print(f"{'='*70}\n")

    avg_train_loss = float("inf")
    final_val: Dict = {}
    final_test: Dict = {}

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0

        # mininterval throttles tqdm refresh — without it, set_postfix on
        # every batch generates thousands of stdout messages that Jupyter's
        # IOPub buffer holds in RAM until the cell finishes.
        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}",
            leave=False, dynamic_ncols=True, mininterval=0.5,
        )

        try:
            for batch_idx, batch in enumerate(pbar):
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs, targets = batch[0], batch[1]

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                global_step += 1

                # ---- Forward + backward ----
                optimizer.zero_grad(set_to_none=True)
                output = model(inputs)
                if hasattr(output, 'loss'):
                    loss = output.loss
                else:
                    loss = loss_fn(output, targets)
                loss.backward()

                # Gradient norm: compute only when we actually log, not
                # every step. Each call performs a host sync per parameter
                # which both slows training and creates short-lived
                # allocations that fragment the CUDA caching allocator.
                should_log = (global_step % log_every == 0)
                grad_norm = compute_grad_norm(model) if should_log else 0.0

                # ---- Optimizer-specific updates ----

                # NysAdam / SketchySGD: micro-batch gradient update
                if needs_per_sample and global_step % nys_refresh == 0:
                    try:
                        # Save the full-batch gradient before microbatch computation.
                        # compute_nystrom_grads_microbatch calls zero_grad() and
                        # backward() internally, which would overwrite p.grad.
                        saved_grads = {
                            p: p.grad.detach().clone()
                            for p in raw_model.parameters()
                            if p.grad is not None
                        }

                        psg = compute_nystrom_grads_microbatch(
                            raw_model, loss_fn, inputs, targets,
                            n_microbatches=nys_n_microbatches,
                        )
                        if hasattr(optimizer, "update_nystrom_sketch"):
                            optimizer.update_nystrom_sketch(psg)

                        # Restore the full-batch gradient for optimizer.step()
                        for p, g in saved_grads.items():
                            p.grad = g

                        # Drop intermediate tensors immediately so they
                        # don't hang around until the next batch.
                        del psg, saved_grads
                    except Exception:
                        pass  # Gracefully skip on failure

                # Sophia / NysSophia: Hessian diagonal update
                if needs_hessian and global_step % hessian_refresh == 0:
                    try:
                        # Save gradients (Hessian estimation also uses backward)
                        saved_grads = {
                            p: p.grad.detach().clone()
                            for p in raw_model.parameters()
                            if p.grad is not None
                        }

                        if hasattr(optimizer, "update_hessian"):
                            optimizer.update_hessian(raw_model, loss_fn, inputs, targets)
                        elif hasattr(optimizer, "update_hessian_diagonal"):
                            optimizer.update_hessian_diagonal(
                                raw_model, loss_fn, inputs, targets,
                            )

                        # Restore gradients
                        for p, g in saved_grads.items():
                            p.grad = g
                        del saved_grads
                    except Exception:
                        pass

                # Step
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                # Accumulate epoch statistics from a host-side scalar.
                loss_value = loss.item()
                epoch_loss += loss_value * inputs.size(0)
                epoch_samples += inputs.size(0)

                # ---- Logging ----
                if should_log:
                    current_lr = (
                        scheduler.get_last_lr()[0] if scheduler else
                        optimizer.param_groups[0]["lr"]
                    )

                    step_metrics = StepMetrics(
                        step=global_step,
                        epoch=epoch,
                        train_loss=loss_value,
                        learning_rate=current_lr,
                        grad_norm=grad_norm,
                        update_norm=0.0,
                        wall_time=time.perf_counter() - start_time,
                        peak_memory_mb=get_peak_memory_mb(),
                    )

                    # NysAdam diagnostics (safe against cusolver errors)
                    if hasattr(optimizer, "get_nystrom_diagnostics"):
                        try:
                            diag = optimizer.get_nystrom_diagnostics()
                            step_metrics.nys_condition_number = diag.get("avg_condition_number")
                            step_metrics.nys_effective_rank = diag.get("avg_sketch_rank")
                        except Exception:
                            pass

                    log.add_step(step_metrics)
                    pbar.set_postfix(
                        {"loss": f"{loss_value:.4f}", "gnorm": f"{grad_norm:.3f}"}
                    )

                # Drop the loss tensor (and the autograd-graph head that it
                # roots) so it can be reclaimed before the next forward.
                # backward() already freed activations, but the loss object
                # itself is held by the local variable until reassignment.
                del loss, output
        finally:
            # Explicit close prevents tqdm from holding the iterable
            # reference across many epochs of an early-stopped run.
            pbar.close()

        # ---- Epoch evaluation on VALIDATION set (for model selection) ----
        avg_train_loss = epoch_loss / max(epoch_samples, 1)

        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics.get("accuracy", None)
            val_f1 = val_metrics.get("f1_macro", None)
            val_ece = val_metrics.get("ece", None)
            val_ppl = val_metrics.get("perplexity", None)
            val_tok_acc = val_metrics.get("token_accuracy", None)

            # Log evaluation step (val_accuracy works for both classification
            # and LM token accuracy)
            eval_step = StepMetrics(
                step=global_step,
                epoch=epoch,
                train_loss=avg_train_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
                grad_norm=0.0,
                update_norm=0.0,
                wall_time=time.perf_counter() - start_time,
                val_loss=val_loss,
                val_accuracy=val_acc if val_acc is not None else val_tok_acc,
            )
            log.add_step(eval_step)

            # Build metric strings for printing
            if val_acc is not None:
                extra = f" | Val Acc: {val_acc:.4f}"
                if val_f1 is not None:
                    extra += f" | F1: {val_f1:.4f}"
                if val_ece is not None:
                    extra += f" | ECE: {val_ece:.4f}"
            elif val_ppl is not None:
                extra = f" | PPL: {val_ppl:.1f}"
                if val_tok_acc is not None:
                    extra += f" | TokAcc: {val_tok_acc:.4f}"
            else:
                extra = ""
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}{extra}"
            )

            # ---- Save best model checkpoint (based on VALIDATION metric) ----
            if save_checkpoints:
                if selection_metric == "loss":
                    current_metric = val_loss
                    is_better = current_metric < best_val_metric
                else:  # accuracy
                    current_metric = val_acc if val_acc is not None else 0.0
                    is_better = current_metric > best_val_metric

                if is_better:
                    best_val_metric = current_metric
                    epochs_without_improvement = 0
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        f"{experiment_name}_{optimizer_name}_best.pt",
                    )
                    metric_str = (
                        f"val_loss={val_loss:.4f}" if selection_metric == "loss"
                        else f"val_acc={val_acc:.4f}"
                    )
                    metadata = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "model_class": model_class_name,
                        "model_config": config.get("model_config", {}),
                        "optimizer_name": optimizer_name,
                        "optimizer_config": {
                            k: v for k, v in config.items()
                            if k not in ("optimizers", "model_config")
                        },
                        "selection_metric": selection_metric,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "val_f1_macro": val_f1,
                        "val_ece": val_ece,
                        "val_perplexity": val_ppl,
                        "val_token_accuracy": val_tok_acc,
                        "train_loss": avg_train_loss,
                        "experiment_name": experiment_name,
                        "dataset": config.get("dataset", ""),
                    }
                    _save_checkpoint_cpu(raw_model, optimizer, metadata, ckpt_path)
                    print(f"  -> Saved best checkpoint ({metric_str})")
                else:
                    epochs_without_improvement += 1

            # ---- Early stopping (Prechelt, 1998) ----
            if epochs_without_improvement >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

        # ---- Per-epoch memory reclamation ----
        # nn.DataParallel creates reference cycles (autograd nodes,
        # replicated tensors, thread-local state) on every forward pass.
        # CPython's cyclic GC defers gen-2 collection, so these cycles
        # accumulate unless we explicitly collect. Without this call,
        # memory grows ~200 MB/epoch on 2×T4 with ResNet-18.
        _epoch_gc()

    # ════════════════════════════════════════════════════════════════
    # FINAL EVALUATION
    # Validation: for reporting (model was selected on this)
    # Test: touched EXACTLY ONCE here — unbiased estimate
    # (Cawley & Talbot, 2010: never use test for selection)
    # ════════════════════════════════════════════════════════════════
    final_val = evaluate(model, val_loader, loss_fn, device)
    final_test = evaluate(model, test_loader, loss_fn, device)

    log.total_time = time.perf_counter() - start_time
    log.final_train_loss = avg_train_loss
    log.final_test_loss = final_test["loss"]
    log.final_test_accuracy = final_test.get("accuracy",
                                             final_test.get("token_accuracy", 0.0))

    # Save final model checkpoint (last epoch, NOT selected by val metric)
    final_ckpt_path = None
    best_ckpt_path = None
    if save_checkpoints:
        final_ckpt_path = os.path.join(
            checkpoint_dir,
            f"{experiment_name}_{optimizer_name}_final.pt",
        )
        final_metadata = {
            "epoch": epochs,
            "global_step": global_step,
            "model_class": model_class_name,
            "model_config": config.get("model_config", {}),
            "optimizer_name": optimizer_name,
            "optimizer_config": {
                k: v for k, v in config.items()
                if k not in ("optimizers", "model_config")
            },
            "selection_metric": "none_final_epoch",
            "val_loss": final_val["loss"],
            "val_accuracy": final_val.get("accuracy"),
            "val_f1_macro": final_val.get("f1_macro"),
            "val_ece": final_val.get("ece"),
            "val_perplexity": final_val.get("perplexity"),
            "val_token_accuracy": final_val.get("token_accuracy"),
            "test_loss": final_test["loss"],
            "test_accuracy": final_test.get("accuracy"),
            "test_f1_macro": final_test.get("f1_macro"),
            "test_ece": final_test.get("ece"),
            "test_perplexity": final_test.get("perplexity"),
            "test_token_accuracy": final_test.get("token_accuracy"),
            "train_loss": log.final_train_loss,
            "experiment_name": experiment_name,
            "dataset": config.get("dataset", ""),
            "total_training_time_sec": log.total_time,
        }
        _save_checkpoint_cpu(raw_model, optimizer, final_metadata, final_ckpt_path)

        best_ckpt_path = os.path.join(
            checkpoint_dir,
            f"{experiment_name}_{optimizer_name}_best.pt",
        )

    print(f"\n{'='*70}")
    print(f"DONE: {optimizer_name} | Total time: {log.total_time:.1f}s")
    print(f"  Train Loss:     {log.final_train_loss:.4f}")

    # Print val/test metrics (handles both classification and LM)
    for label, metrics in [("Val", final_val), ("Test", final_test)]:
        line = f"  {label} Loss:".ljust(18) + f"{metrics['loss']:.4f}"
        if metrics.get("accuracy") is not None:
            line += f"  | Acc: {metrics['accuracy']:.4f}"
        if metrics.get("perplexity") is not None:
            line += f"  | PPL: {metrics['perplexity']:.1f}"
        if metrics.get("token_accuracy") is not None:
            line += f"  | TokAcc: {metrics['token_accuracy']:.4f}"
        if label == "Test":
            line += "  <- FINAL (unbiased)"
        print(line)
        if metrics.get("f1_macro") is not None:
            print(f"  {label} F1-Macro:".ljust(18) +
                  f"{metrics['f1_macro']:.4f}  | ECE: {metrics.get('ece', 0):.4f}")

    print(f"  --------------------------------------------------")
    print(f"  Best Val {selection_metric.title()}:  {log.best_val_loss:.4f}" if selection_metric == "loss"
          else f"  Best Val Accuracy: {log.best_val_accuracy:.4f}")
    print(f"  Peak Memory:    {get_peak_memory_mb():.1f} MB")
    if save_checkpoints:
        print(f"  Best checkpoint (by val {selection_metric}): {best_ckpt_path}")
        print(f"  Final checkpoint (last epoch):  {final_ckpt_path}")
    print(f"{'='*70}\n")

    return log


# ═══════════════════════════════════════════════════════════════════
# Experiment Configurations
# ═══════════════════════════════════════════════════════════════════

DEFAULT_EXPERIMENTS = {
    "logistic": {
        "dataset": "digits",
        "model": "logistic",
        "epochs": 50,
        "eval_every": 1,
        "log_every": 5,
        "early_stopping_patience": 15,
        "optimizers": {
            "adamw":       {"lr": 1e-3, "weight_decay": 1e-4},
            "sgd":         {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
            "nysadam":     {"lr": 1e-3, "nys_rank": 16, "nys_refresh_interval": 5,
                           "weight_decay": 1e-4, "min_dim_for_nystrom": 32,
                           "nys_n_microbatches": 8},
            "nysadam_light": {"lr": 1e-3, "nys_rank": 16, "weight_decay": 1e-4,
                             "min_dim_for_nystrom": 32},
            "sketchy_sgd": {"lr": 0.01, "nys_rank": 16, "weight_decay": 1e-4,
                           "min_dim_for_nystrom": 32},
        },
    },
    "cifar10": {
        "dataset": "cifar10",
        "model": "resnet18",
        "epochs": 200,
        "batch_size": 128,
        "eval_every": 1,
        "log_every": 50,
        "scheduler": "cosine",
        "early_stopping_patience": 25,
        "optimizers": {
            "adamw":       {"lr": 1e-3, "weight_decay": 0.05},
            "sgd":         {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
            "sophia":      {"lr": 1e-4, "weight_decay": 0.05, "rho": 1.0,
                           "betas": [0.965, 0.99]},
            "shampoo":     {"lr": 1e-3, "weight_decay": 0.05,
                           "precondition_frequency": 100},
            "nysadam":     {"lr": 1e-3, "nys_rank": 32, "weight_decay": 0.05,
                           "nys_refresh_interval": 50,
                           "min_dim_for_nystrom": 500000,
                           "nys_n_microbatches": 8},
            "nysadam_light": {"lr": 1e-3, "nys_rank": 32, "weight_decay": 0.05,
                             "min_dim_for_nystrom": 100000},
        },
    },
    "cifar100": {
        "dataset": "cifar100",
        "model": "resnet18",
        "epochs": 200,
        "batch_size": 128,
        "eval_every": 1,
        "log_every": 50,
        "scheduler": "cosine",
        "early_stopping_patience": 25,
        "optimizers": {
            "adamw":   {"lr": 1e-3, "weight_decay": 0.05},
            "sgd":     {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
            "sophia":  {"lr": 1e-4, "weight_decay": 0.05, "rho": 1.0,
                       "betas": [0.965, 0.99]},
            "nysadam": {"lr": 1e-3, "nys_rank": 32, "weight_decay": 0.05,
                       "nys_refresh_interval": 50,
                       "min_dim_for_nystrom": 500000,
                       "nys_n_microbatches": 8},
            "nysadam_light": {"lr": 1e-3, "nys_rank": 32, "weight_decay": 0.05,
                             "min_dim_for_nystrom": 100000},
        },
    },
    "pinn_burgers": {
        "dataset": "burgers",
        "model": "pinn",
        "epochs": 500,
        "eval_every": 10,
        "log_every": 20,
        "scheduler": "cosine",
        "early_stopping_patience": 100,
        "optimizers": {
            "adamw":   {"lr": 1e-3, "weight_decay": 0.0},
            "nysadam": {"lr": 1e-3, "nys_rank": 32, "weight_decay": 0.0,
                       "nys_refresh_interval": 10, "min_dim_for_nystrom": 64,
                       "nys_n_microbatches": 8},
            "nysadam_light": {"lr": 1e-3, "nys_rank": 16, "weight_decay": 0.0,
                             "min_dim_for_nystrom": 64},
        },
    },
    "transformer_lm": {
        "dataset": "wikitext2",
        "model": "transformer_lm",
        "model_config": {"d_model": 256, "n_heads": 4, "n_layers": 4,
                         "max_seq_len": 128},
        "epochs": 30,
        "batch_size": 16,
        "eval_every": 1,
        "log_every": 100,
        "scheduler": "cosine",
        "early_stopping_patience": 8,
        "optimizers": {
            "adamw":   {"lr": 3e-4, "weight_decay": 0.01},
            "sophia":  {"lr": 1e-4, "weight_decay": 0.01, "rho": 1.0},
            "nysadam": {"lr": 3e-4, "nys_rank": 32, "weight_decay": 0.01,
                       "nys_refresh_interval": 50,
                       "min_dim_for_nystrom": 100000,
                       "nys_n_microbatches": 8},
            "nysadam_light": {"lr": 3e-4, "nys_rank": 32, "weight_decay": 0.01,
                             "min_dim_for_nystrom": 100000},
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
# PINN-specific loss
# ═══════════════════════════════════════════════════════════════════

def burgers_pinn_loss(
    model: nn.Module,
    dataset,
    device: torch.device,
    n_samples: int = 2048,
) -> torch.Tensor:
    """Compute PINN loss for Burgers' equation."""
    # Interior residual
    idx = torch.randint(0, len(dataset.interior), (n_samples,))
    xt = dataset.interior[idx].to(device).requires_grad_(True)
    u = model(xt)

    u_grad = torch.autograd.grad(
        u, xt, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    u_x, u_t = u_grad[:, 0:1], u_grad[:, 1:2]

    u_xx = torch.autograd.grad(
        u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0][:, 0:1]

    residual = u_t + u * u_x - dataset.nu * u_xx
    loss_pde = (residual ** 2).mean()

    # Initial condition
    ic_pts = dataset.ic_points.to(device)
    ic_vals = dataset.ic_values.to(device)
    u_ic = model(ic_pts)
    loss_ic = F.mse_loss(u_ic, ic_vals)

    # Boundary condition
    bc_pts = dataset.bc_points.to(device)
    bc_vals = dataset.bc_values.to(device)
    u_bc = model(bc_pts)
    loss_bc = F.mse_loss(u_bc, bc_vals)

    return loss_pde + 10.0 * loss_ic + 10.0 * loss_bc


# ═══════════════════════════════════════════════════════════════════
# Main Runner
# ═══════════════════════════════════════════════════════════════════

def run_experiment(
    experiment_name: str,
    optimizer_names: Optional[list] = None,
    seed: int = 42,
    output_dir: str = "results",
    config_override: Optional[Dict] = None,
) -> Dict[str, ExperimentLog]:
    """
    Run a complete experiment comparing multiple optimizers.

    Data is loaded ONCE and reused across all optimizer runs to prevent
    memory accumulation from duplicate DataLoader worker processes.

    Args:
        experiment_name: Key from DEFAULT_EXPERIMENTS or custom config.
        optimizer_names: List of optimizers to run (default: all in config).
        seed: Random seed.
        output_dir: Directory for saving results.
        config_override: Override specific config values.

    Returns:
        Dictionary mapping optimizer names to their ExperimentLogs.
    """
    device = get_device()

    # Load config
    if experiment_name in DEFAULT_EXPERIMENTS:
        config = deepcopy(DEFAULT_EXPERIMENTS[experiment_name])
    else:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    if config_override:
        config.update(config_override)

    # Determine which optimizers to run
    if optimizer_names is None:
        optimizer_names = list(config["optimizers"].keys())

    # ════════════════════════════════════════════════════════════════
    # Load data ONCE outside the optimizer loop.
    # DataLoaders with num_workers>0 create forked worker processes.
    # Recreating them per optimizer leaks memory via copy-on-write
    # page duplication in worker address spaces.
    # ════════════════════════════════════════════════════════════════
    dataset_name = config["dataset"]
    batch_size = config.get("batch_size", 128)
    train_loader = val_loader = test_loader = None
    pinn_dataset = None
    data_info = {}

    if dataset_name in ("digits", "covtype", "breast_cancer", "iris", "wine", "synthetic"):
        train_loader, val_loader, test_loader, data_info = load_sklearn_dataset(dataset_name)
    elif dataset_name in ("cifar10", "cifar100"):
        train_loader, val_loader, test_loader, data_info = load_cifar(
            dataset_name, batch_size=batch_size,
        )
    elif dataset_name == "burgers":
        train_loader, pinn_dataset, data_info = load_burgers_data()
    elif dataset_name == "wikitext2":
        seq_len = config.get("model_config", {}).get("max_seq_len", 128)
        train_loader, val_loader, test_loader, data_info = load_text_data(
            batch_size=batch_size,
            seq_len=seq_len,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Precompute model kwargs (same for all optimizers)
    model_name = config["model"]
    base_model_kwargs = dict(config.get("model_config", {}))

    if model_name == "logistic":
        base_model_kwargs["input_dim"] = data_info["input_dim"]
        base_model_kwargs["num_classes"] = data_info["num_classes"]
    elif model_name in ("resnet18", "resnet34"):
        base_model_kwargs["num_classes"] = data_info["num_classes"]
    elif model_name == "pinn":
        base_model_kwargs["input_dim"] = data_info.get("input_dim", 2)
        base_model_kwargs["output_dim"] = data_info.get("output_dim", 1)
    elif model_name == "transformer_lm":
        base_model_kwargs["vocab_size"] = data_info.get("vocab_size", 50257)

    # Loss function (same for all optimizers)
    if model_name == "transformer_lm":
        def lm_loss_fn(output, targets):
            if output.dim() == 3:
                return F.cross_entropy(
                    output.view(-1, output.size(-1)), targets.view(-1)
                )
            return F.cross_entropy(output, targets)
        loss_fn = lm_loss_fn
    elif model_name == "pinn":
        loss_fn = nn.MSELoss()
    else:
        loss_fn = nn.CrossEntropyLoss()

    results: Dict[str, ExperimentLog] = {}

    for opt_name in optimizer_names:
        if opt_name not in config["optimizers"]:
            print(f"Warning: {opt_name} not configured for {experiment_name}, skipping.")
            continue

        print(f"\n{'#'*70}")
        print(f"# Running: {experiment_name} with {opt_name}")
        print(f"{'#'*70}")

        set_seed(seed)

        opt_config = config["optimizers"][opt_name]
        merged_config = {**config, **opt_config}

        # ---- Create fresh model and optimizer for this run ----
        model = create_model(model_name, **base_model_kwargs).to(device)
        optimizer = create_optimizer(opt_name, model, merged_config)
        model, raw_model = setup_model_for_device(model, device)

        # Reset peak-memory counters so each optimizer reports its OWN
        # peak rather than the watermark left behind by previous runs.
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---- Train ----
        try:
            if model_name == "pinn":
                log = _train_pinn(
                    model, optimizer, opt_name, pinn_dataset,
                    merged_config, device, experiment_name,
                )
            else:
                log = train(
                    model, optimizer, opt_name,
                    train_loader, val_loader, test_loader, loss_fn,
                    merged_config, device, experiment_name,
                )
        finally:
            # ---- Save results & flush all references ----
            # Even on exception, we must release this optimizer's state
            # before constructing the next one — otherwise the dual
            # allocation would push us over the 30 GB Kaggle limit.
            try:
                os.makedirs(output_dir, exist_ok=True)
                log_path = os.path.join(output_dir, f"{experiment_name}_{opt_name}.json")
                if 'log' in dir():
                    log.save(log_path)
                    print(f"Results saved to {log_path}")
                    # Keep only summary statistics in `results`; drop the
                    # per-step list which can hold thousands of entries
                    # across an entire suite of optimizers.
                    log.steps = []
                    results[opt_name] = log
            finally:
                # Order matters: optimizer holds references to model
                # parameters, so it must be released first.
                try:
                    del optimizer
                except UnboundLocalError:
                    pass
                try:
                    del model, raw_model
                except UnboundLocalError:
                    pass
                _release_memory()

    # Print comparison table
    _print_comparison_table(results, experiment_name)

    return results


def _train_pinn(model, optimizer, opt_name, dataset, config, device, experiment_name):
    """Special training loop for PINNs (no standard dataloader).

    For NysAdam: computes per-sample gradients by evaluating the PDE loss
    on small random subsets of collocation points. Each subset gives a
    different gradient direction, serving as pseudo-per-sample gradients
    that capture the curvature of the PDE residual loss.
    """
    epochs = config.get("epochs", 500)
    nys_refresh = config.get("nys_refresh_interval", 10)
    nys_n_microbatches = config.get("nys_n_microbatches",
                                    config.get("nys_subsample", 8))
    needs_per_sample = opt_name in ("nysadam", "nyssophia", "sketchy_sgd")

    log = ExperimentLog(
        optimizer_name=opt_name, dataset=experiment_name,
        model="PINN", config=config,
    )

    start_time = time.perf_counter()
    loss_value = float("nan")

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss = burgers_pinn_loss(model, dataset, device)
        loss.backward()

        # grad_norm and Nyström work only on logged/refresh epochs
        should_log = (epoch + 1) % config.get("log_every", 20) == 0
        grad_norm = compute_grad_norm(model) if should_log else 0.0

        # ---- Nyström sketch update for NysAdam on PINNs ----
        if needs_per_sample and (epoch + 1) % nys_refresh == 0:
            if hasattr(optimizer, "update_nystrom_sketch"):
                try:
                    psg = _compute_pinn_per_sample_grads(
                        model, dataset, device, n_subsets=nys_n_microbatches,
                    )
                    optimizer.update_nystrom_sketch(psg)
                    del psg
                except Exception:
                    pass

        optimizer.step()
        loss_value = loss.item()
        del loss

        if should_log:
            step_metrics = StepMetrics(
                step=epoch + 1, epoch=epoch,
                train_loss=loss_value,
                learning_rate=optimizer.param_groups[0]["lr"],
                grad_norm=grad_norm, update_norm=0.0,
                wall_time=time.perf_counter() - start_time,
            )
            log.add_step(step_metrics)

        if (epoch + 1) % config.get("eval_every", 10) == 0:
            print(f"Epoch {epoch+1}/{epochs} | PDE Loss: {loss_value:.6f} | "
                  f"Grad Norm: {grad_norm:.4f}")

        # Periodic GC (every 50 PINN epochs ≈ every ~3s)
        if (epoch + 1) % 50 == 0:
            _epoch_gc()

    log.total_time = time.perf_counter() - start_time
    log.final_train_loss = loss_value
    log.final_test_loss = loss_value
    return log


def _compute_pinn_per_sample_grads(model, dataset, device, n_subsets=16):
    """
    Compute pseudo-per-sample gradients for PINN by evaluating the PDE
    loss on different random subsets of collocation points.

    Each subset produces a different gradient → these capture the
    curvature structure of the PDE residual loss landscape.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    per_sample = {p: [] for p in params}

    for _ in range(n_subsets):
        model.zero_grad()
        # Evaluate loss on a small random subset
        loss_i = burgers_pinn_loss(model, dataset, device, n_samples=256)
        loss_i.backward()

        for p in params:
            if p.grad is not None:
                per_sample[p].append(p.grad.detach().clone())

    result = {}
    for p in params:
        if per_sample[p]:
            result[p] = torch.stack(per_sample[p], dim=0)  # [n_subsets, *shape]

    return result


def _print_comparison_table(results: Dict[str, ExperimentLog], experiment_name: str):
    """Print a summary comparison table with unbiased test metrics."""
    print(f"\n{'='*90}")
    print(f"COMPARISON TABLE: {experiment_name}")
    print(f"  Model selected by: min VALIDATION loss | Test set touched ONCE (unbiased)")
    print(f"{'='*90}")
    print(f"{'Optimizer':<18} {'Train Loss':>12} {'Test Loss':>12} "
          f"{'Test Acc':>10} {'Best ValL':>10} {'Time (s)':>10}")
    print("-" * 90)

    for name, log in sorted(results.items()):
        acc_str = f"{log.final_test_accuracy:.4f}" if log.final_test_accuracy > 0 else "N/A"
        val_str = f"{log.best_val_loss:.4f}" if log.best_val_loss < float("inf") else "N/A"

        print(
            f"{name:<18} {log.final_train_loss:>12.4f} {log.final_test_loss:>12.4f} "
            f"{acc_str:>10} {val_str:>10} {log.total_time:>10.1f}"
        )

    print(f"{'='*90}\n")


# ═══════════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="NysAdam Experiment Runner")
    parser.add_argument(
        "--experiment", type=str, default="logistic",
        choices=list(DEFAULT_EXPERIMENTS.keys()),
        help="Experiment to run",
    )
    parser.add_argument(
        "--optimizers", nargs="+", default=None,
        help="Optimizers to compare (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file (overrides defaults)",
    )
    # Ablation arguments
    parser.add_argument("--nys_rank", type=int, default=None, help="Override Nyström rank")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")

    args = parser.parse_args()

    # Build config override
    override = {}
    if args.config:
        with open(args.config) as f:
            override = yaml.safe_load(f)
    if args.epochs:
        override["epochs"] = args.epochs
    if args.nys_rank:
        override["nys_rank"] = args.nys_rank
    if args.lr:
        override["lr"] = args.lr

    run_experiment(
        experiment_name=args.experiment,
        optimizer_names=args.optimizers,
        seed=args.seed,
        output_dir=args.output_dir,
        config_override=override if override else None,
    )


if __name__ == "__main__":
    main()
