# NysAdam: Nyström-Augmented Adaptive Moment Estimation

**A Regularized Nyström-Preconditioned Optimizer for Non-Convex Optimization in Machine Learning, Deep Learning, and Large Language Models**

**Author:** Hardik Tankaria  
**Date:** May 2026  

---

## Abstract

NysAdam is the first optimizer that augments Adam's per-parameter adaptivity with a rank-*m* Nyström approximation of cross-parameter curvature from the empirical Fisher information matrix. It smoothly interpolates between AdamW (*m*=0) and the natural gradient (*m*→*d*), with convergence guarantees that explicitly depend on the Nyström rank and the effective dimension of the curvature. We demonstrate state-of-the-art efficiency on logistic regression, deep learning (CIFAR-10/100, ImageNet), physics-informed neural networks (PINNs), and LLM fine-tuning.

---

## Project Structure

```
nysadam/
├── README.md                          # This file
├── setup.py                           # Package installation
├── requirements.txt                   # Dependencies
├── configs/                           # Experiment configurations
│   ├── logistic_regression.yaml
│   ├── cifar10_resnet.yaml
│   ├── cifar100_resnet.yaml
│   ├── pinn_burgers.yaml
│   └── llm_finetune.yaml
├── nysadam/                           # Main package
│   ├── __init__.py
│   ├── optimizers/                    # All optimizer implementations
│   │   ├── __init__.py
│   │   ├── nysadam.py                 # ★ NysAdam (proposed method)
│   │   ├── nysadam_variants.py        # NysSophia, NysAdam-Light, etc.
│   │   ├── baselines.py              # AdamW, SGD wrappers with logging
│   │   ├── sophia.py                  # Sophia optimizer (baseline)
│   │   ├── shampoo.py                 # Shampoo optimizer (baseline)
│   │   └── sketchy_sgd.py            # SketchySGD (baseline)
│   ├── models/                        # Model architectures
│   │   ├── __init__.py
│   │   ├── logistic.py               # Logistic regression
│   │   ├── resnet.py                  # ResNet-18/32/110
│   │   ├── vit.py                     # Vision Transformer (DeiT-S)
│   │   ├── pinn.py                    # Physics-Informed Neural Network
│   │   └── transformer_lm.py         # Transformer language model
│   ├── data/                          # Data loading utilities
│   │   ├── __init__.py
│   │   ├── classical.py              # UCI/LIBSVM datasets
│   │   ├── vision.py                  # CIFAR-10/100, ImageNet
│   │   ├── pinn_data.py              # PDE collocation points
│   │   └── text.py                    # Text data for LLM experiments
│   ├── utils/                         # Utilities
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Loss, accuracy, timing metrics
│   │   ├── logging_utils.py          # Experiment logging
│   │   ├── hessian_analysis.py       # Hessian spectrum analysis tools
│   │   ├── nystrom_utils.py          # Nyström math utilities
│   │   └── reproducibility.py        # Seeds, deterministic settings
│   └── experiments/                   # Experiment runners
│       ├── __init__.py
│       ├── run_logistic.py           # Exp 1: Logistic regression
│       ├── run_cifar.py              # Exp 2: CIFAR-10/100
│       ├── run_pinn.py               # Exp 3: PINNs
│       ├── run_llm_finetune.py       # Exp 4: LLM fine-tuning
│       └── run_all.py                # Master experiment runner
├── scripts/                           # Shell scripts for experiments
│   ├── run_all_experiments.sh
│   ├── run_ablation_rank.sh          # Ablation: varying Nyström rank m
│   └── run_ablation_refresh.sh       # Ablation: varying refresh interval k
├── tests/                             # Unit tests
│   ├── test_nysadam.py               # Optimizer correctness tests
│   ├── test_woodbury.py              # Numerical stability tests
│   └── test_nystrom_utils.py         # Nyström math tests
└── results/                           # Output directory (auto-populated)
    ├── figures/
    └── tables/
```

---

## Quick Start

```bash
# Install
pip install -e .

# Run all experiments
python -m nysadam.experiments.run_all --config configs/cifar10_resnet.yaml

# Run specific experiment
python -m nysadam.experiments.run_cifar --optimizer nysadam --rank 64

# Run ablation study
bash scripts/run_ablation_rank.sh
```

---

## Key Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Learning rate | η | 1e-3 | Step size |
| Nyström rank | m | 64 | Rank of curvature approximation (0 = AdamW) |
| Refresh interval | k | 10 | Steps between Nyström updates |
| Sketch decay | β_nys | 0.999 | EMA decay for Nyström sketch |
| Momentum decay | β₁ | 0.9 | First moment EMA (same as Adam) |
| Second moment decay | β₂ | 0.999 | Second moment EMA (same as Adam) |
| Clipping threshold | ρ | 1.0 | Element-wise update clipping |
| Weight decay | λ_wd | 0.01 | Decoupled weight decay |
| Epsilon | ε | 1e-8 | Adam denominator stability |

---


## License

MIT License
