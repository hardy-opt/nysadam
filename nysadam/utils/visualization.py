"""
Visualization and Plotting for NysAdam Paper Figures
====================================================
Author: Hardik Tankaria
Date: May 2026
Description:
    Publication-quality figures for IEEE TPAMI submission.
    Generates:
    - Training curves (loss & accuracy vs steps/epochs)
    - Wall-clock convergence comparison
    - Ablation plots (rank m, refresh interval k)
    - Hessian spectrum / effective dimension plots
    - Comparison tables (LaTeX-formatted)

Usage:
    python -m nysadam.utils.visualization --results_dir results/ --output_dir results/figures/
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from glob import glob

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ═══════════════════════════════════════════════════════════════════
# Style Configuration (IEEE TPAMI compatible)
# ═══════════════════════════════════════════════════════════════════

OPTIMIZER_STYLES = {
    "nysadam":       {"color": "#D62728", "marker": "o", "linestyle": "-",  "label": "NysAdam (ours)"},
    "nysadam_light": {"color": "#FF7F0E", "marker": "s", "linestyle": "-",  "label": "NysAdam-Light (ours)"},
    "nyssophia":     {"color": "#E377C2", "marker": "D", "linestyle": "-",  "label": "NysSophia (ours)"},
    "adamw":         {"color": "#1F77B4", "marker": "^", "linestyle": "--", "label": "AdamW"},
    "sgd":           {"color": "#2CA02C", "marker": "v", "linestyle": "--", "label": "SGD"},
    "sophia":        {"color": "#9467BD", "marker": "P", "linestyle": "-.", "label": "Sophia"},
    "shampoo":       {"color": "#8C564B", "marker": "X", "linestyle": "-.", "label": "Shampoo"},
    "sketchy_sgd":   {"color": "#7F7F7F", "marker": "d", "linestyle": ":", "label": "SketchySGD"},
}

FIGSIZE_SINGLE = (5, 3.5)
FIGSIZE_DOUBLE = (10, 3.5)
FIGSIZE_QUAD = (10, 7)
DPI = 300
FONT_SIZE = 10


def _setup_style():
    """Configure matplotlib for publication quality."""
    if not HAS_MPL:
        return
    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "font.family": "serif",
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE + 1,
        "legend.fontsize": FONT_SIZE - 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
    })


def _get_style(opt_name: str) -> dict:
    """Get plot style for an optimizer."""
    return OPTIMIZER_STYLES.get(opt_name, {
        "color": "#333333", "marker": ".", "linestyle": "-", "label": opt_name,
    })


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def load_experiment_logs(results_dir: str, experiment_name: str) -> Dict[str, dict]:
    """
    Load all experiment logs for a given experiment name.

    Args:
        results_dir: Directory containing JSON result files.
        experiment_name: Experiment prefix (e.g., 'cifar10').

    Returns:
        Dict mapping optimizer names to their log data.
    """
    logs = {}
    pattern = os.path.join(results_dir, f"{experiment_name}_*.json")
    for path in sorted(glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        opt_name = data.get("optimizer_name", os.path.basename(path).split("_")[-1].replace(".json", ""))
        logs[opt_name] = data
    return logs


def _extract_series(log_data: dict, x_key: str, y_key: str):
    """Extract x, y series from log steps, skipping None values."""
    xs, ys = [], []
    for step in log_data.get("steps", []):
        x_val = step.get(x_key)
        y_val = step.get(y_key)
        if x_val is not None and y_val is not None:
            xs.append(x_val)
            ys.append(y_val)
    return np.array(xs), np.array(ys)


# ═══════════════════════════════════════════════════════════════════
# Figure 1: Training Loss Curves
# ═══════════════════════════════════════════════════════════════════

def plot_training_curves(
    logs: Dict[str, dict],
    experiment_name: str,
    output_dir: str,
    x_axis: str = "step",    # 'step', 'epoch', 'wall_time'
    y_axis: str = "train_loss",
    log_scale_y: bool = True,
) -> str:
    """
    Plot training loss/accuracy curves for all optimizers.

    Args:
        logs: Optimizer name → log data.
        experiment_name: For title and filename.
        output_dir: Save directory.
        x_axis: X-axis variable.
        y_axis: Y-axis variable.
        log_scale_y: Use log scale on Y axis.

    Returns:
        Path to saved figure.
    """
    if not HAS_MPL:
        print("matplotlib not available, skipping plot")
        return ""

    _setup_style()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for opt_name, log_data in logs.items():
        style = _get_style(opt_name)
        xs, ys = _extract_series(log_data, x_axis, y_axis)

        if len(xs) == 0:
            continue

        # Subsample markers for clarity
        marker_every = max(1, len(xs) // 15)
        ax.plot(
            xs, ys,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=marker_every,
            label=style["label"],
        )

    if log_scale_y:
        ax.set_yscale("log")

    x_labels = {"step": "Training Steps", "epoch": "Epoch", "wall_time": "Wall-Clock Time (s)"}
    y_labels = {
        "train_loss": "Training Loss",
        "val_loss": "Validation Loss",
        "val_accuracy": "Validation Accuracy",
    }
    ax.set_xlabel(x_labels.get(x_axis, x_axis))
    ax.set_ylabel(y_labels.get(y_axis, y_axis))
    ax.set_title(f"{experiment_name}: {y_labels.get(y_axis, y_axis)}")
    ax.legend(loc="best", framealpha=0.9)

    os.makedirs(output_dir, exist_ok=True)
    fname = f"{experiment_name}_{y_axis}_vs_{x_axis}.pdf"
    path = os.path.join(output_dir, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# Figure 2: Test Accuracy / Loss Curves
# ═══════════════════════════════════════════════════════════════════

def plot_test_curves(
    logs: Dict[str, dict],
    experiment_name: str,
    output_dir: str,
) -> List[str]:
    """Plot both test loss and test accuracy (if available)."""
    paths = []

    # Test loss
    p = plot_training_curves(logs, experiment_name, output_dir,
                             x_axis="epoch", y_axis="val_loss", log_scale_y=True)
    if p:
        paths.append(p)

    # Test accuracy (check if any log has it)
    has_acc = any(
        any(s.get("val_accuracy") is not None for s in d.get("steps", []))
        for d in logs.values()
    )
    if has_acc:
        p = plot_training_curves(logs, experiment_name, output_dir,
                                 x_axis="epoch", y_axis="val_accuracy", log_scale_y=False)
        if p:
            paths.append(p)

    return paths


# ═══════════════════════════════════════════════════════════════════
# Figure 3: Wall-Clock Convergence
# ═══════════════════════════════════════════════════════════════════

def plot_wallclock_convergence(
    logs: Dict[str, dict],
    experiment_name: str,
    output_dir: str,
) -> str:
    """Plot training loss vs wall-clock time for fair time comparison."""
    return plot_training_curves(
        logs, experiment_name, output_dir,
        x_axis="wall_time", y_axis="train_loss", log_scale_y=True,
    )


# ═══════════════════════════════════════════════════════════════════
# Figure 4: Ablation — Nyström Rank m
# ═══════════════════════════════════════════════════════════════════

def plot_rank_ablation(
    results_dir: str,
    experiment_name: str,
    output_dir: str,
    metric: str = "final_test_loss",
) -> str:
    """
    Plot final performance vs Nyström rank m.

    Expects result files named: {experiment}_nysadam_rank{m}.json
    """
    if not HAS_MPL:
        return ""

    _setup_style()
    ranks = []
    values = []

    for path in sorted(glob(os.path.join(results_dir, f"{experiment_name}_nysadam_rank*.json"))):
        with open(path) as f:
            data = json.load(f)
        # Extract rank from filename
        fname = os.path.basename(path)
        rank = int(fname.split("rank")[1].replace(".json", ""))
        ranks.append(rank)
        values.append(data.get(metric, float("nan")))

    if not ranks:
        print(f"No rank ablation data found for {experiment_name}")
        return ""

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(ranks, values, "o-", color=OPTIMIZER_STYLES["nysadam"]["color"],
            linewidth=2, markersize=8, label="NysAdam")

    # Add AdamW baseline if available
    adamw_path = os.path.join(results_dir, f"{experiment_name}_adamw.json")
    if os.path.exists(adamw_path):
        with open(adamw_path) as f:
            adamw_data = json.load(f)
        adamw_val = adamw_data.get(metric, None)
        if adamw_val is not None:
            ax.axhline(y=adamw_val, color=OPTIMIZER_STYLES["adamw"]["color"],
                       linestyle="--", linewidth=1.5, label="AdamW (m=0)")

    ax.set_xlabel("Nyström Rank m")
    y_label = metric.replace("_", " ").title()
    ax.set_ylabel(y_label)
    ax.set_title(f"{experiment_name}: {y_label} vs Nyström Rank")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{experiment_name}_rank_ablation.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# Figure 5: Hessian Spectrum
# ═══════════════════════════════════════════════════════════════════

def plot_hessian_spectrum(
    eigenvalues: np.ndarray,
    experiment_name: str,
    output_dir: str,
    effective_dims: Optional[Dict[str, float]] = None,
) -> str:
    """Plot Hessian eigenvalue spectrum with effective dimension markers."""
    if not HAS_MPL:
        return ""

    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)

    # Left: Eigenvalue spectrum (log scale)
    indices = np.arange(1, len(eigenvalues) + 1)
    ax1.semilogy(indices, np.sort(np.abs(eigenvalues))[::-1], "b-", linewidth=1.5)
    ax1.set_xlabel("Eigenvalue Index")
    ax1.set_ylabel("Eigenvalue Magnitude")
    ax1.set_title("Hessian Eigenvalue Spectrum")

    # Right: Cumulative energy (fraction of trace)
    sorted_eigs = np.sort(np.abs(eigenvalues))[::-1]
    cumulative = np.cumsum(sorted_eigs) / np.sum(sorted_eigs)
    ax2.plot(indices, cumulative, "r-", linewidth=1.5)
    ax2.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5, label="90% energy")
    ax2.axhline(y=0.99, color="gray", linestyle=":", alpha=0.5, label="99% energy")

    # Mark effective dimension
    if effective_dims:
        for lam_str, d_eff in effective_dims.items():
            ax2.axvline(x=d_eff, color="green", linestyle="--", alpha=0.5,
                       label=f"d_eff({lam_str})={d_eff:.0f}")

    ax2.set_xlabel("Number of Eigenvalues")
    ax2.set_ylabel("Cumulative Energy Fraction")
    ax2.set_title("Spectral Energy Concentration")
    ax2.legend(fontsize=FONT_SIZE - 2)

    fig.suptitle(f"{experiment_name}: Hessian Spectrum Analysis", fontsize=FONT_SIZE + 1)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{experiment_name}_hessian_spectrum.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# Figure 6: Combined Multi-Panel Figure (for paper)
# ═══════════════════════════════════════════════════════════════════

def plot_combined_figure(
    all_logs: Dict[str, Dict[str, dict]],
    output_dir: str,
) -> str:
    """
    Create a multi-panel figure with all experiments.

    Args:
        all_logs: experiment_name → {optimizer_name → log_data}
    """
    if not HAS_MPL:
        return ""

    _setup_style()
    experiments = list(all_logs.keys())
    n_exp = len(experiments)
    fig, axes = plt.subplots(2, n_exp, figsize=(4 * n_exp, 7))

    if n_exp == 1:
        axes = axes.reshape(2, 1)

    for col, exp_name in enumerate(experiments):
        logs = all_logs[exp_name]

        # Top row: Training loss vs steps
        ax_top = axes[0, col]
        for opt_name, log_data in logs.items():
            style = _get_style(opt_name)
            xs, ys = _extract_series(log_data, "step", "train_loss")
            if len(xs) > 0:
                marker_every = max(1, len(xs) // 10)
                ax_top.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                           marker=style["marker"], markevery=marker_every,
                           label=style["label"])
        ax_top.set_yscale("log")
        ax_top.set_title(exp_name)
        ax_top.set_xlabel("Steps")
        if col == 0:
            ax_top.set_ylabel("Training Loss")

        # Bottom row: Test loss/acc vs epochs
        ax_bot = axes[1, col]
        y_key = "val_accuracy"
        has_acc = any(
            any(s.get(y_key) is not None for s in d.get("steps", []))
            for d in logs.values()
        )
        if not has_acc:
            y_key = "val_loss"

        for opt_name, log_data in logs.items():
            style = _get_style(opt_name)
            xs, ys = _extract_series(log_data, "epoch", y_key)
            if len(xs) > 0:
                marker_every = max(1, len(xs) // 10)
                ax_bot.plot(xs, ys, color=style["color"], linestyle=style["linestyle"],
                           marker=style["marker"], markevery=marker_every,
                           label=style["label"])

        if y_key == "val_loss":
            ax_bot.set_yscale("log")
        ax_bot.set_xlabel("Epoch")
        if col == 0:
            ylabel = "Test Accuracy" if y_key == "val_accuracy" else "Test Loss"
            ax_bot.set_ylabel(ylabel)

    # Single legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4),
              bbox_to_anchor=(0.5, -0.02), fontsize=FONT_SIZE - 1)

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "combined_results.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# Tables: LaTeX-formatted comparison
# ═══════════════════════════════════════════════════════════════════

def generate_comparison_table(
    all_logs: Dict[str, Dict[str, dict]],
    output_dir: str,
) -> str:
    """
    Generate LaTeX table comparing all optimizers across experiments.

    Returns path to .tex file.
    """
    experiments = list(all_logs.keys())
    all_optimizers = set()
    for logs in all_logs.values():
        all_optimizers.update(logs.keys())
    all_optimizers = sorted(all_optimizers)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comparison of optimizers across benchmarks. "
        r"\textbf{Bold} = best, \underline{underline} = second best.}",
        r"\label{tab:main_results}",
        r"\small",
    ]

    # Header
    n_exp = len(experiments)
    col_spec = "l" + "cc" * n_exp
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    header = "Optimizer"
    for exp in experiments:
        header += f" & \\multicolumn{{2}}{{c}}{{{exp}}}"
    header += r" \\"
    lines.append(header)

    subheader = ""
    for _ in experiments:
        subheader += " & Test Loss & Test Acc"
    subheader += r" \\"
    lines.append(r"\cmidrule(lr){2-" + str(1 + 2 * n_exp) + "}")
    lines.append(subheader)
    lines.append(r"\midrule")

    # Data rows
    for opt in all_optimizers:
        style = _get_style(opt)
        row = style["label"]

        for exp in experiments:
            logs = all_logs.get(exp, {})
            if opt in logs:
                data = logs[opt]
                tl = data.get("final_test_loss", float("nan"))
                ta = data.get("final_test_accuracy", 0)
                row += f" & {tl:.4f} & {ta:.2%}" if ta > 0 else f" & {tl:.4f} & --"
            else:
                row += " & -- & --"

        row += r" \\"
        lines.append(row)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "comparison_table.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════
# Master Visualization Runner
# ═══════════════════════════════════════════════════════════════════

def generate_all_figures(results_dir: str, output_dir: str):
    """Generate all paper figures from experiment results."""
    experiments = ["logistic", "cifar10", "cifar100", "pinn_burgers", "transformer_lm"]

    all_logs = {}
    for exp in experiments:
        logs = load_experiment_logs(results_dir, exp)
        if logs:
            all_logs[exp] = logs
            print(f"\n--- Generating figures for: {exp} ---")
            plot_training_curves(logs, exp, output_dir, x_axis="step", y_axis="train_loss")
            plot_test_curves(logs, exp, output_dir)
            plot_wallclock_convergence(logs, exp, output_dir)
            plot_rank_ablation(results_dir, exp, output_dir)

    if all_logs:
        print("\n--- Generating combined figure ---")
        plot_combined_figure(all_logs, output_dir)
        print("\n--- Generating LaTeX table ---")
        generate_comparison_table(all_logs, os.path.join(output_dir, "..", "tables"))


def main():
    parser = argparse.ArgumentParser(description="Generate NysAdam paper figures")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/figures")
    args = parser.parse_args()
    generate_all_figures(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
