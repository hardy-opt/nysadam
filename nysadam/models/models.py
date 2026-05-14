"""
Model Architectures
===================
Author: Hardik Tankaria
Date: May 2026
Description:
    Model implementations for NysAdam benchmarking experiments.
    Covers: logistic regression, MLP, ResNet, and PINN models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════════
# Logistic Regression (Experiment 1: Classical ML)
# ═══════════════════════════════════════════════════════════════════

class LogisticRegression(nn.Module):
    """
    Linear logistic regression model.
    Used for: validating NysAdam on convex problems, comparing with
    SketchySGD/Nys-Newton baselines, confirming m=0 ≡ Adam.
    """

    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron with ReLU activations.
    Used for: small-scale non-convex validation experiments.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        num_classes: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(dropout)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.view(x.size(0), -1))


# ═══════════════════════════════════════════════════════════════════
# ResNet (Experiment 2: CIFAR-10/100, ImageNet)
# ═══════════════════════════════════════════════════════════════════

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    """
    ResNet for CIFAR and ImageNet.
    Supports ResNet-18, ResNet-32, ResNet-110.
    """

    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet18(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def resnet34(num_classes=10, **kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# PINN (Experiment 3: Physics-Informed Neural Networks)
# ═══════════════════════════════════════════════════════════════════

class PINN(nn.Module):
    """
    Physics-Informed Neural Network for solving PDEs.

    Standard MLP with tanh activation (smooth for derivative computation).
    Used for Burgers' equation, heat equation, etc.

    The PINN loss requires automatic differentiation through the network
    to compute PDE residuals — making curvature-aware optimizers
    particularly beneficial due to the pathological loss landscapes.
    """

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_dims: list = [64, 64, 64, 64],
        activation: str = "tanh",
    ):
        super().__init__()
        act_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}[activation]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), act_fn()])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Xavier initialization (important for PINNs)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════
# Simple Transformer Language Model (Experiment 4: LLM)
# ═══════════════════════════════════════════════════════════════════

class TransformerLM(nn.Module):
    """
    Small transformer language model for optimizer benchmarking.
    Designed to be trainable on a single GPU for fair comparison.

    Configurations:
        - Small:  d_model=256,  n_layers=4,  n_heads=4   (~5M params)
        - Medium: d_model=512,  n_layers=6,  n_heads=8   (~25M params)
        - Large:  d_model=768,  n_layers=12, n_heads=12  (~85M params)
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            return loss

        return logits


# ═══════════════════════════════════════════════════════════════════
# Factory function
# ═══════════════════════════════════════════════════════════════════

def create_model(name: str, **kwargs) -> nn.Module:
    """Create model by name."""
    models = {
        "logistic": LogisticRegression,
        "mlp": MLP,
        "resnet18": lambda **kw: resnet18(**kw),
        "resnet34": lambda **kw: resnet34(**kw),
        "pinn": PINN,
        "transformer_lm": TransformerLM,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](**kwargs)
