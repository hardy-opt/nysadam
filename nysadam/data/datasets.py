"""
Data Loading Utilities
======================
Author: Hardik Tankaria
Date: May 2026
Description:
    Unified data loading for all experiment types:
    - Classical ML: sklearn datasets, LIBSVM format
    - Vision: CIFAR-10/100, ImageNet
    - PINNs: Collocation point generation
    - Text: WikiText, OpenWebText for LM experiments
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple, Optional


# ═══════════════════════════════════════════════════════════════════
# Classical ML Datasets
# ═══════════════════════════════════════════════════════════════════

def load_sklearn_dataset(
    name: str = "digits",
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load a classical ML dataset from sklearn with proper 3-way split.

    Following Hastie, Tibshirani & Friedman (2009, Ch. 7):
    - Training set: for parameter updates
    - Validation set: for model selection / checkpoint selection
    - Test set: touched ONCE for final unbiased evaluation

    Args:
        name: Dataset name ('digits', 'covtype', 'breast_cancer',
              'iris', 'wine', 'synthetic').
        val_size: Fraction for validation split.
        test_size: Fraction for test split.
        seed: Random seed.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn import datasets

    if name == "digits":
        data = datasets.load_digits()
        X, y = data.data, data.target
    elif name == "breast_cancer":
        data = datasets.load_breast_cancer()
        X, y = data.data, data.target
    elif name == "iris":
        data = datasets.load_iris()
        X, y = data.data, data.target
    elif name == "wine":
        data = datasets.load_wine()
        X, y = data.data, data.target
    elif name == "covtype":
        data = datasets.fetch_covtype()
        X, y = data.data[:50000], data.target[:50000] - 1
    elif name == "synthetic":
        X, y = datasets.make_classification(
            n_samples=10000, n_features=100, n_informative=50,
            n_classes=10, random_state=seed,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3-way split: train → (train + val) / test, then train → train / val
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y,
    )
    relative_val = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=relative_val,
        random_state=seed, stratify=y_trainval,
    )

    # To tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train), batch_size=128, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=256, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test), batch_size=256, shuffle=False,
    )

    info = {
        "input_dim": X_train.shape[1],
        "num_classes": len(np.unique(y)),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }

    return train_loader, val_loader, test_loader, info


# ═══════════════════════════════════════════════════════════════════
# Vision Datasets
# ═══════════════════════════════════════════════════════════════════

def load_cifar(
    dataset: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    augment: bool = True,
    val_fraction: float = 0.1,
    pin_memory: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load CIFAR-10 or CIFAR-100 with proper train/val/test split.

    The original 50k training set is split into:
    - 45k training (with augmentation)
    - 5k validation (NO augmentation, clean evaluation)
    The 10k test set is held out and touched only once.

    Uses num_workers=0 (in-process loading) because CIFAR images are
    32x32 and transforms take microseconds per batch. Multiprocessing
    workers cause copy-on-write memory growth on Linux due to Python
    GC modifying reference counts in forked address spaces.

    pin_memory defaults to False on hosts with constrained CPU RAM
    (e.g. Kaggle's 30 GB limit shared with the dataset). Pinned host
    pages are never released back to the OS by PyTorch's allocator and
    accumulate as the CUDA driver creates staging buffers for transfers,
    which we observed pushing process RSS past the limit during
    multi-optimizer comparison runs. With non_blocking=True transfers
    from regular pageable memory still overlap forward/backward
    reasonably well on a single-GPU setup.

    Args:
        dataset: 'cifar10' or 'cifar100'.
        batch_size: Batch size.
        num_workers: Ignored (kept for API compatibility). Always uses 0.
        data_dir: Download directory.
        augment: Apply data augmentation to training set.
        val_fraction: Fraction of training data for validation.
        pin_memory: Whether to pin host memory. Defaults to False to keep
            CPU RAM usage predictable on memory-constrained hosts.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
    """
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import Subset

    if pin_memory is None:
        pin_memory = False

    if dataset == "cifar10":
        DatasetClass = torchvision.datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 10
    elif dataset == "cifar100":
        DatasetClass = torchvision.datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]) if augment else T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    transform_eval = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    # Load each split exactly once (no duplicate datasets)
    full_train = DatasetClass(data_dir, train=True, download=True, transform=transform_train)
    test_set = DatasetClass(data_dir, train=False, download=True, transform=transform_eval)

    # Deterministic train/val split
    n_total = len(full_train)
    n_val = int(n_total * val_fraction)
    gen = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=gen).tolist()
    train_indices = indices[:n_total - n_val]
    val_indices = indices[n_total - n_val:]

    train_set = Subset(full_train, train_indices)

    # Validation subset: wrap with a transform-overriding dataset
    # so validation images get clean (no augment) transforms without
    # loading the entire dataset a second time
    val_set = _SubsetWithTransform(full_train, val_indices, transform_eval)

    # num_workers=0: CIFAR images are 32x32, transforms are trivial.
    # Workers cause copy-on-write memory growth of 500+ MB per epoch
    # on Linux due to Python GC modifying refcounts in forked address
    # spaces. In-process loading adds <1% overhead for CIFAR.
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=pin_memory,
    )

    info = {
        "num_classes": num_classes,
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
        "input_channels": 3,
        "image_size": 32,
    }

    return train_loader, val_loader, test_loader, info


class _SubsetWithTransform(Dataset):
    """
    A subset of a dataset that overrides the transform.

    This avoids loading the dataset twice (once with augmentation,
    once without). The validation subset uses the parent dataset's
    raw data but applies a clean (no augmentation) transform.
    """

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Access the raw PIL image and target from the parent dataset
        # CIFAR datasets store images as numpy arrays in self.data
        # and labels in self.targets
        real_idx = self.indices[idx]
        from PIL import Image
        img = Image.fromarray(self.dataset.data[real_idx])
        target = self.dataset.targets[real_idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# ═══════════════════════════════════════════════════════════════════
# PINN Collocation Points
# ═══════════════════════════════════════════════════════════════════

class BurgersDataset(Dataset):
    """
    Collocation points for Burgers' equation:
        u_t + u * u_x = ν * u_xx
    on domain x ∈ [-1, 1], t ∈ [0, 1].

    Boundary conditions:
        u(x, 0) = -sin(π x)     (initial condition)
        u(-1, t) = u(1, t) = 0  (boundary conditions)
    """

    def __init__(
        self,
        n_interior: int = 10000,
        n_boundary: int = 1000,
        n_initial: int = 1000,
        nu: float = 0.01 / np.pi,
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)

        # Interior collocation points
        x_int = rng.uniform(-1, 1, (n_interior, 1))
        t_int = rng.uniform(0, 1, (n_interior, 1))
        self.interior = torch.tensor(
            np.hstack([x_int, t_int]), dtype=torch.float32
        )

        # Initial condition points (t=0)
        x_ic = rng.uniform(-1, 1, (n_initial, 1))
        t_ic = np.zeros((n_initial, 1))
        u_ic = -np.sin(np.pi * x_ic)
        self.ic_points = torch.tensor(
            np.hstack([x_ic, t_ic]), dtype=torch.float32
        )
        self.ic_values = torch.tensor(u_ic, dtype=torch.float32)

        # Boundary condition points (x=±1)
        t_bc = rng.uniform(0, 1, (n_boundary, 1))
        x_bc_left = -np.ones((n_boundary // 2, 1))
        x_bc_right = np.ones((n_boundary // 2, 1))
        t_bc_left = t_bc[:n_boundary // 2]
        t_bc_right = t_bc[n_boundary // 2:]
        self.bc_points = torch.tensor(
            np.vstack([
                np.hstack([x_bc_left, t_bc_left]),
                np.hstack([x_bc_right, t_bc_right]),
            ]),
            dtype=torch.float32,
        )
        self.bc_values = torch.zeros(n_boundary, 1)
        self.nu = nu

    def __len__(self):
        return len(self.interior)

    def __getitem__(self, idx):
        return self.interior[idx], torch.zeros(1)  # Dummy target


def load_burgers_data(
    n_interior: int = 10000,
    batch_size: int = 1024,
    **kwargs,
) -> Tuple[DataLoader, "BurgersDataset", dict]:
    """
    Load Burgers' equation PINN data.

    Returns:
        (train_loader, dataset, info_dict)
    """
    dataset = BurgersDataset(n_interior=n_interior, **kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    info = {
        "input_dim": 2,
        "output_dim": 1,
        "n_interior": n_interior,
        "nu": dataset.nu,
    }
    return loader, dataset, info


# ═══════════════════════════════════════════════════════════════════
# Text Data for Language Modeling
# ═══════════════════════════════════════════════════════════════════

class SimpleTextDataset(Dataset):
    """
    Token-level text dataset for language model training.

    Uses non-overlapping chunks of seq_len tokens, which is the standard
    approach for LM training (GPT, GPT-2, etc.). A sliding window with
    stride 1 would create N-seq_len samples from N tokens (massive
    redundancy and impractical training times).

    Non-overlapping: N tokens / seq_len = number of sequences.
    For WikiText-2 (~2.4M tokens, seq_len=128): ~18,750 sequences.
    """

    def __init__(
        self,
        data: torch.Tensor,
        seq_len: int = 128,
    ):
        self.seq_len = seq_len
        # Trim data to exact multiple of seq_len + 1 (for targets)
        n_chunks = len(data) // (seq_len + 1)
        self.data = data[:n_chunks * (seq_len + 1)]
        self.n_chunks = n_chunks

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * (self.seq_len + 1)
        chunk = self.data[start:start + self.seq_len + 1]
        x = chunk[:-1]  # input tokens
        y = chunk[1:]   # target tokens (shifted by 1)
        return x, y


def load_text_data(
    dataset_name: str = "wikitext2",
    seq_len: int = 256,
    batch_size: int = 32,
    data_dir: str = "./data",
) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    """
    Load text data for language modeling experiments.

    WikiText-2 natively provides train/validation/test splits.
    Tokenization is done per-line to avoid triggering the HuggingFace
    tokenizer warning about exceeding model_max_length (the tokenizer
    warns when a single encode() call exceeds GPT-2's 1024-token limit,
    even though our SimpleTextDataset chunks the result into seq_len
    blocks afterward).

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
    """
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1")

        def _tokenize_split(split_texts):
            """Tokenize per-line and concatenate to avoid length warning."""
            all_ids = []
            for line in split_texts:
                if line.strip():  # skip empty lines
                    all_ids.extend(tokenizer.encode(line))
            return torch.tensor(all_ids, dtype=torch.long)

        train_tokens = _tokenize_split(ds["train"]["text"])
        val_tokens = _tokenize_split(ds["validation"]["text"])
        test_tokens = _tokenize_split(ds["test"]["text"])

        vocab_size = tokenizer.vocab_size

    except (ImportError, Exception):
        # Fallback: synthetic data for testing
        vocab_size = 5000
        train_tokens = torch.randint(0, vocab_size, (100000,))
        val_tokens = torch.randint(0, vocab_size, (10000,))
        test_tokens = torch.randint(0, vocab_size, (10000,))

    train_dataset = SimpleTextDataset(train_tokens, seq_len)
    val_dataset = SimpleTextDataset(val_tokens, seq_len)
    test_dataset = SimpleTextDataset(test_tokens, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    info = {
        "vocab_size": vocab_size,
        "seq_len": seq_len,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "test_tokens": len(test_tokens),
    }

    return train_loader, val_loader, test_loader, info
