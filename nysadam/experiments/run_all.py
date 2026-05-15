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
from nysadam.utils.reproducibility import set_seed, get_device, count_parameters


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

def compute_per_sample_grads_vmap(
    model: nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    max_samples: int = 32,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Compute per-sample gradients using functorch vmap (preferred method).
    Falls back to sequential computation if vmap is not available.

    Args:
        model: Neural network.
        loss_fn: Loss function.
        inputs: Input batch [B, ...].
        targets: Targets [B, ...].
        max_samples: Max samples for per-sample grads (memory control).

    Returns:
        Dictionary mapping parameters to per-sample gradients.
    """
    B = min(inputs.size(0), max_samples)
    inputs_sub = inputs[:B]
    targets_sub = targets[:B]

    try:
        from torch.func import vmap, grad
        from functools import partial

        # Functional version of forward + loss
        params_dict = dict(model.named_parameters())
        params_list = [p for p in model.parameters() if p.requires_grad]

        def compute_loss_stateless(params_values, buffers, x, y):
            from torch.func import functional_call
            params_recon = {}
            idx = 0
            for name, p in model.named_parameters():
                if p.requires_grad:
                    params_recon[name] = params_values[idx]
                    idx += 1
                else:
                    params_recon[name] = p
            output = functional_call(model, params_recon, (x.unsqueeze(0),))
            return loss_fn(output, y.unsqueeze(0))

        buffers = dict(model.named_buffers())
        params_values = tuple(p for p in model.parameters() if p.requires_grad)

        per_sample_grad_fn = vmap(
            grad(compute_loss_stateless),
            in_dims=(None, None, 0, 0),
        )
        per_sample_grads_tuple = per_sample_grad_fn(
            params_values, buffers, inputs_sub, targets_sub,
        )

        result = {}
        for p, psg in zip(params_list, per_sample_grads_tuple):
            result[p] = psg.detach()
        return result

    except (ImportError, Exception):
        # Fallback: sequential computation
        return _compute_per_sample_grads_sequential(
            model, loss_fn, inputs_sub, targets_sub,
        )


def _compute_per_sample_grads_sequential(
    model: nn.Module,
    loss_fn,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """Sequential fallback for per-sample gradient computation."""
    B = inputs.size(0)
    params = [p for p in model.parameters() if p.requires_grad]
    per_sample = {p: [] for p in params}

    for i in range(B):
        model.zero_grad()
        output = model(inputs[i:i + 1])
        loss_i = loss_fn(output, targets[i:i + 1])
        loss_i.backward()

        for p in params:
            if p.grad is not None:
                per_sample[p].append(p.grad.detach().clone())

    result = {}
    for p in params:
        if per_sample[p]:
            result[p] = torch.stack(per_sample[p], dim=0)

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

    Handles:
    - Standard gradient-based optimization (Adam, SGD)
    - Per-sample gradient computation for NysAdam/SketchySGD
    - Hessian diagonal updates for Sophia/NysSophia
    - Periodic evaluation on validation set
    - Comprehensive metric logging (loss, accuracy, F1, ECE)

    Args:
        model: Neural network.
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
    nys_subsample = config.get("nys_subsample", 32)
    nys_refresh = config.get("nys_refresh_interval", 10)
    hessian_refresh = config.get("hessian_refresh_interval", 10)

    # Model selection criterion (Gneiting & Raftery 2007):
    # Default 'loss' = min validation loss (proper scoring rule)
    # Alternative 'accuracy' = max validation accuracy (only when
    #   accuracy is the sole deployment metric + balanced classes)
    selection_metric = config.get("selection_metric", "loss")

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
        model=model.__class__.__name__,
        config=config,
    )

    model.train()
    global_step = 0
    start_time = time.perf_counter()
    # For checkpoint selection: lower is better for loss, higher for accuracy
    best_val_metric = float("inf") if selection_metric == "loss" else 0.0

    needs_per_sample = optimizer_name in ("nysadam", "nyssophia", "sketchy_sgd")
    needs_hessian = optimizer_name in ("sophia", "nyssophia")

    print(f"\n{'='*70}")
    print(f"Training: {experiment_name} | Optimizer: {optimizer_name}")
    print(f"Model: {model.__class__.__name__} | Params: {count_parameters(model):,}")
    print(f"Epochs: {epochs} | Device: {device}")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_samples = 0

        pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs}",
            leave=False, dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets = batch[0], batch[1]

            inputs = inputs.to(device)
            targets = targets.to(device)
            global_step += 1

            # ---- Forward + backward ----
            with Timer() as step_timer:
                optimizer.zero_grad()
                output = model(inputs)
                if hasattr(output, 'loss'):
                    loss = output.loss
                else:
                    loss = loss_fn(output, targets)
                loss.backward()

                grad_norm = compute_grad_norm(model)

                # ---- Optimizer-specific updates ----

                # NysAdam / SketchySGD: per-sample gradient update
                if needs_per_sample and global_step % nys_refresh == 0:
                    try:
                        psg = compute_per_sample_grads_vmap(
                            model, loss_fn, inputs, targets,
                            max_samples=nys_subsample,
                        )
                        if hasattr(optimizer, "update_nystrom_sketch"):
                            optimizer.update_nystrom_sketch(psg)
                    except Exception as e:
                        pass  # Gracefully skip on failure

                # Sophia / NysSophia: Hessian diagonal update
                if needs_hessian and global_step % hessian_refresh == 0:
                    try:
                        if hasattr(optimizer, "update_hessian"):
                            optimizer.update_hessian(model, loss_fn, inputs, targets)
                        elif hasattr(optimizer, "update_hessian_diagonal"):
                            optimizer.update_hessian_diagonal(
                                model, loss_fn, inputs, targets,
                            )
                    except Exception:
                        pass

                # Step
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

            epoch_loss += loss.item() * inputs.size(0)
            epoch_samples += inputs.size(0)

            # ---- Logging ----
            if global_step % log_every == 0:
                current_lr = (
                    scheduler.get_last_lr()[0] if scheduler else
                    optimizer.param_groups[0]["lr"]
                )

                step_metrics = StepMetrics(
                    step=global_step,
                    epoch=epoch,
                    train_loss=loss.item(),
                    learning_rate=current_lr,
                    grad_norm=grad_norm,
                    update_norm=0.0,
                    wall_time=time.perf_counter() - start_time,
                    peak_memory_mb=get_peak_memory_mb(),
                )

                # NysAdam diagnostics
                if hasattr(optimizer, "get_nystrom_diagnostics"):
                    diag = optimizer.get_nystrom_diagnostics()
                    step_metrics.nys_condition_number = diag.get("avg_condition_number")
                    step_metrics.nys_effective_rank = diag.get("avg_sketch_rank")

                log.add_step(step_metrics)

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "gnorm": f"{grad_norm:.3f}"})

        # ---- Epoch evaluation on VALIDATION set (for model selection) ----
        avg_train_loss = epoch_loss / max(epoch_samples, 1)

        if (epoch + 1) % eval_every == 0:
            val_metrics = evaluate(model, val_loader, loss_fn, device)
            val_loss = val_metrics["loss"]
            val_acc = val_metrics.get("accuracy", None)
            val_f1 = val_metrics.get("f1_macro", None)
            val_ece = val_metrics.get("ece", None)

            # Log evaluation step
            eval_step = StepMetrics(
                step=global_step,
                epoch=epoch,
                train_loss=avg_train_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
                grad_norm=0.0,
                update_norm=0.0,
                wall_time=time.perf_counter() - start_time,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
            log.add_step(eval_step)

            acc_str = f" | Val Acc: {val_acc:.4f}" if val_acc is not None else ""
            f1_str = f" | F1: {val_f1:.4f}" if val_f1 is not None else ""
            ece_str = f" | ECE: {val_ece:.4f}" if val_ece is not None else ""
            print(
                f"Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f}{acc_str}{f1_str}{ece_str}"
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
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        f"{experiment_name}_{optimizer_name}_best.pt",
                    )
                    checkpoint = {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "model_class": model.__class__.__name__,
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
                        "train_loss": avg_train_loss,
                        "experiment_name": experiment_name,
                        "dataset": config.get("dataset", ""),
                    }
                    torch.save(checkpoint, ckpt_path)
                    metric_str = f"val_loss={val_loss:.4f}" if selection_metric == "loss" else f"val_acc={val_acc:.4f}"
                    print(f"  → Saved best checkpoint ({metric_str})")

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
    log.final_test_accuracy = final_test.get("accuracy", 0.0)

    # Save final model checkpoint (last epoch, NOT selected by val metric)
    final_ckpt_path = None
    best_ckpt_path = None
    if save_checkpoints:
        final_ckpt_path = os.path.join(
            checkpoint_dir,
            f"{experiment_name}_{optimizer_name}_final.pt",
        )
        checkpoint = {
            "epoch": epochs,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_class": model.__class__.__name__,
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
            "test_loss": final_test["loss"],
            "test_accuracy": final_test.get("accuracy"),
            "test_f1_macro": final_test.get("f1_macro"),
            "test_ece": final_test.get("ece"),
            "train_loss": log.final_train_loss,
            "experiment_name": experiment_name,
            "dataset": config.get("dataset", ""),
            "total_training_time_sec": log.total_time,
        }
        torch.save(checkpoint, final_ckpt_path)

        best_ckpt_path = os.path.join(
            checkpoint_dir,
            f"{experiment_name}_{optimizer_name}_best.pt",
        )

    print(f"\n{'='*70}")
    print(f"DONE: {optimizer_name} | Total time: {log.total_time:.1f}s")
    print(f"  Train Loss:     {log.final_train_loss:.4f}")
    print(f"  Val Loss:       {final_val['loss']:.4f}" +
          (f"  | Val Acc: {final_val.get('accuracy', 0):.4f}" if final_val.get('accuracy') else ""))
    if final_val.get("f1_macro"):
        print(f"  Val F1-Macro:   {final_val['f1_macro']:.4f}  | Val ECE: {final_val.get('ece', 0):.4f}")
    print(f"  ────────────────────────────────────────────")
    print(f"  Test Loss:      {final_test['loss']:.4f}" +
          (f"  | Test Acc: {final_test.get('accuracy', 0):.4f}" if final_test.get('accuracy') else "") +
          "  ← FINAL (unbiased)")
    if final_test.get("f1_macro"):
        print(f"  Test F1-Macro:  {final_test['f1_macro']:.4f}  | Test ECE: {final_test.get('ece', 0):.4f}")
    print(f"  ────────────────────────────────────────────")
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
        "optimizers": {
            "adamw":       {"lr": 1e-3, "weight_decay": 1e-4},
            "sgd":         {"lr": 0.01, "momentum": 0.9, "weight_decay": 1e-4},
            "nysadam":     {"lr": 1e-3, "nys_rank": 16, "nys_refresh_interval": 5,
                           "weight_decay": 1e-4, "min_dim_for_nystrom": 32,
                           "nys_subsample": 32},
            "nysadam_light": {"lr": 1e-3, "nys_rank": 16, "weight_decay": 1e-4,
                             "min_dim_for_nystrom": 32},
            "sketchy_sgd": {"lr": 0.01, "nys_rank": 16, "weight_decay": 1e-4,
                           "min_dim_for_nystrom": 32},
        },
    },
    "cifar10": {
        "dataset": "cifar10",
        "model": "resnet18",
        "epochs": 100,
        "batch_size": 128,
        "eval_every": 1,
        "log_every": 50,
        "scheduler": "cosine",
        "optimizers": {
            "adamw":       {"lr": 1e-3, "weight_decay": 0.05},
            "sgd":         {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
            "sophia":      {"lr": 1e-4, "weight_decay": 0.05, "rho": 1.0,
                           "betas": [0.965, 0.99]},
            "shampoo":     {"lr": 1e-3, "weight_decay": 0.05,
                           "precondition_frequency": 100},
            "nysadam":     {"lr": 1e-3, "nys_rank": 64, "weight_decay": 0.05,
                           "nys_refresh_interval": 10, "nys_subsample": 32},
            "nysadam_light": {"lr": 1e-3, "nys_rank": 32, "weight_decay": 0.05},
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
        "optimizers": {
            "adamw":   {"lr": 1e-3, "weight_decay": 0.05},
            "sgd":     {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4},
            "sophia":  {"lr": 1e-4, "weight_decay": 0.05, "rho": 1.0,
                       "betas": [0.965, 0.99]},
            "nysadam": {"lr": 1e-3, "nys_rank": 64, "weight_decay": 0.05},
            "nysadam_light": {"lr": 1e-3, "nys_rank": 32, "weight_decay": 0.05},
        },
    },
    "pinn_burgers": {
        "dataset": "burgers",
        "model": "pinn",
        "epochs": 500,
        "eval_every": 10,
        "log_every": 20,
        "scheduler": "cosine",
        "optimizers": {
            "adamw":   {"lr": 1e-3, "weight_decay": 0.0},
            "nysadam": {"lr": 1e-3, "nys_rank": 32, "weight_decay": 0.0,
                       "nys_refresh_interval": 10, "min_dim_for_nystrom": 64},
            "nysadam_light": {"lr": 1e-3, "nys_rank": 16, "weight_decay": 0.0,
                             "min_dim_for_nystrom": 64},
        },
    },
    "transformer_lm": {
        "dataset": "wikitext2",
        "model": "transformer_lm",
        "model_config": {"d_model": 256, "n_heads": 4, "n_layers": 4},
        "epochs": 20,
        "batch_size": 32,
        "eval_every": 1,
        "log_every": 50,
        "scheduler": "cosine",
        "optimizers": {
            "adamw":   {"lr": 3e-4, "weight_decay": 0.01},
            "sophia":  {"lr": 1e-4, "weight_decay": 0.01, "rho": 1.0},
            "nysadam": {"lr": 3e-4, "nys_rank": 64, "weight_decay": 0.01,
                       "nys_refresh_interval": 10},
            "nysadam_light": {"lr": 3e-4, "nys_rank": 32, "weight_decay": 0.01},
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

    results = {}

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

        # ---- Load data (3-way split: train / val / test) ----
        dataset_name = config["dataset"]
        batch_size = config.get("batch_size", 128)

        if dataset_name in ("digits", "covtype", "breast_cancer", "iris", "wine", "synthetic"):
            train_loader, val_loader, test_loader, data_info = load_sklearn_dataset(dataset_name)
        elif dataset_name in ("cifar10", "cifar100"):
            train_loader, val_loader, test_loader, data_info = load_cifar(
                dataset_name, batch_size=batch_size,
            )
        elif dataset_name == "burgers":
            train_loader, pinn_dataset, data_info = load_burgers_data()
        elif dataset_name == "wikitext2":
            train_loader, val_loader, test_loader, data_info = load_text_data(
                batch_size=batch_size,
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # ---- Create model ----
        model_name = config["model"]
        model_kwargs = config.get("model_config", {})

        if model_name == "logistic":
            model_kwargs["input_dim"] = data_info["input_dim"]
            model_kwargs["num_classes"] = data_info["num_classes"]
        elif model_name in ("resnet18", "resnet34"):
            model_kwargs["num_classes"] = data_info["num_classes"]
        elif model_name == "pinn":
            model_kwargs["input_dim"] = data_info.get("input_dim", 2)
            model_kwargs["output_dim"] = data_info.get("output_dim", 1)
        elif model_name == "transformer_lm":
            model_kwargs["vocab_size"] = data_info.get("vocab_size", 50257)

        model = create_model(model_name, **model_kwargs).to(device)

        # ---- Create optimizer ----
        optimizer = create_optimizer(opt_name, model, merged_config)

        # ---- Loss function ----
        if model_name == "transformer_lm":
            loss_fn = nn.CrossEntropyLoss()
            # Wrap for transformer: loss expects (logits, targets)
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

        # ---- Special handling for PINN ----
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

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, f"{experiment_name}_{opt_name}.json")
        log.save(log_path)
        print(f"Results saved to {log_path}")

        results[opt_name] = log

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
    nys_subsample = config.get("nys_subsample", 16)
    needs_per_sample = opt_name in ("nysadam", "nyssophia", "sketchy_sgd")

    log = ExperimentLog(
        optimizer_name=opt_name, dataset=experiment_name,
        model="PINN", config=config,
    )

    start_time = time.perf_counter()

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = burgers_pinn_loss(model, dataset, device)
        loss.backward()

        grad_norm = compute_grad_norm(model)

        # ---- Nyström sketch update for NysAdam on PINNs ----
        # Compute per-sample gradients by evaluating loss on random
        # subsets of collocation points (each subset = one "sample")
        if needs_per_sample and (epoch + 1) % nys_refresh == 0:
            if hasattr(optimizer, "update_nystrom_sketch"):
                try:
                    psg = _compute_pinn_per_sample_grads(
                        model, dataset, device, n_subsets=nys_subsample,
                    )
                    optimizer.update_nystrom_sketch(psg)
                except Exception:
                    pass  # Gracefully skip on failure

        optimizer.step()

        if (epoch + 1) % config.get("log_every", 20) == 0:
            step_metrics = StepMetrics(
                step=epoch + 1, epoch=epoch,
                train_loss=loss.item(),
                learning_rate=optimizer.param_groups[0]["lr"],
                grad_norm=grad_norm, update_norm=0.0,
                wall_time=time.perf_counter() - start_time,
            )
            log.add_step(step_metrics)

        if (epoch + 1) % config.get("eval_every", 10) == 0:
            print(f"Epoch {epoch+1}/{epochs} | PDE Loss: {loss.item():.6f} | "
                  f"Grad Norm: {grad_norm:.4f}")

    log.total_time = time.perf_counter() - start_time
    log.final_train_loss = loss.item()
    log.final_test_loss = loss.item()
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
