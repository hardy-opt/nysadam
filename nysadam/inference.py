"""
NysAdam: Model Inference Script
================================
Author: Hardik Tankaria
Date: May 2026
Description:
    Load a trained checkpoint and run inference. Works for all model types
    (LogisticRegression, ResNet, PINN, TransformerLM).

Usage:
    # Classify a CIFAR-10 image
    python -m nysadam.inference --checkpoint results/checkpoints/cifar10_nysadam_best.pt --input image.png

    # Run PINN prediction on a grid
    python -m nysadam.inference --checkpoint results/checkpoints/pinn_burgers_nysadam_best.pt --pinn_grid 100

    # Evaluate checkpoint on test set
    python -m nysadam.inference --checkpoint results/checkpoints/cifar10_nysadam_best.pt --evaluate
"""

import os
import sys
import argparse
import json
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nysadam.models.models import create_model
from nysadam.utils.reproducibility import get_device


def load_checkpoint(checkpoint_path: str, device: Optional[torch.device] = None):
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Device to load model onto (default: auto-detect).

    Returns:
        (model, checkpoint_metadata) tuple.
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model from saved config
    model_class = checkpoint["model_class"]
    model_config = checkpoint.get("model_config", {})
    state_dict = checkpoint["model_state_dict"]

    # Handle checkpoints saved from DataParallel-wrapped models:
    # 1. Strip "module." prefix from state dict keys if present
    # 2. Infer actual model class from state dict structure
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        checkpoint["model_state_dict"] = state_dict

    if model_class == "DataParallel":
        # Infer actual class from state dict keys
        if any("transformer" in k for k in state_dict.keys()):
            model_class = "TransformerLM"
        elif any("layer4" in k or "layer3" in k for k in state_dict.keys()):
            model_class = "ResNet"
        elif any("network" in k for k in state_dict.keys()):
            # Could be PINN or MLP - check input dim
            first_weight = [v for k, v in state_dict.items() if "weight" in k][0]
            model_class = "PINN" if first_weight.shape[1] == 2 else "MLP"
        else:
            model_class = "LogisticRegression"

    # Infer model constructor args from checkpoint
    if model_class == "LogisticRegression":
        # Infer dimensions from state dict
        weight_key = [k for k in checkpoint["model_state_dict"] if "weight" in k][0]
        weight_shape = checkpoint["model_state_dict"][weight_key].shape
        model_config.setdefault("num_classes", weight_shape[0])
        model_config.setdefault("input_dim", weight_shape[1])

    elif model_class == "ResNet":
        linear_key = [k for k in checkpoint["model_state_dict"] if "linear.weight" in k][0]
        num_classes = checkpoint["model_state_dict"][linear_key].shape[0]
        model_config.setdefault("num_classes", num_classes)

    elif model_class == "PINN":
        model_config.setdefault("input_dim", 2)
        model_config.setdefault("output_dim", 1)
        model_config.setdefault("hidden_dims", [64, 64, 64, 64])

    elif model_class == "TransformerLM":
        model_config.setdefault("d_model", 256)
        model_config.setdefault("n_heads", 4)
        model_config.setdefault("n_layers", 4)

    # Create and load model
    model_name_map = {
        "LogisticRegression": "logistic",
        "ResNet": "resnet18",
        "PINN": "pinn",
        "TransformerLM": "transformer_lm",
    }
    model_name = model_name_map.get(model_class, model_class.lower())
    model = create_model(model_name, **model_config)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Metadata
    metadata = {
        "model_class": model_class,
        "model_config": model_config,
        "optimizer": checkpoint.get("optimizer_name", "unknown"),
        "experiment": checkpoint.get("experiment_name", "unknown"),
        "dataset": checkpoint.get("dataset", "unknown"),
        "epoch": checkpoint.get("epoch", -1),
        "selection_metric": checkpoint.get("selection_metric", "loss"),
        "val_loss": checkpoint.get("val_loss", None),
        "val_accuracy": checkpoint.get("val_accuracy", None),
        "val_f1_macro": checkpoint.get("val_f1_macro", None),
        "val_ece": checkpoint.get("val_ece", None),
        "test_loss": checkpoint.get("test_loss", None),
        "test_accuracy": checkpoint.get("test_accuracy", None),
        "training_time_sec": checkpoint.get("total_training_time_sec", None),
    }

    return model, metadata


@torch.no_grad()
def predict_image(model, image_tensor: torch.Tensor, class_names=None):
    """
    Classify a single image.

    Args:
        model: Trained classification model.
        image_tensor: Preprocessed image tensor [1, C, H, W] or [C, H, W].
        class_names: Optional list of class name strings.

    Returns:
        Dict with 'class_id', 'class_name', 'confidence', 'top5'.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    logits = model(image_tensor)
    probs = F.softmax(logits, dim=1)[0]

    top5_probs, top5_ids = torch.topk(probs, min(5, probs.shape[0]))

    result = {
        "class_id": top5_ids[0].item(),
        "confidence": top5_probs[0].item(),
        "top5": [
            {
                "class_id": top5_ids[i].item(),
                "class_name": class_names[top5_ids[i].item()] if class_names else str(top5_ids[i].item()),
                "confidence": top5_probs[i].item(),
            }
            for i in range(len(top5_ids))
        ],
    }

    if class_names:
        result["class_name"] = class_names[top5_ids[0].item()]

    return result


@torch.no_grad()
def predict_pinn(model, x_grid: np.ndarray, t_grid: np.ndarray):
    """
    Evaluate PINN on a space-time grid.

    Args:
        model: Trained PINN model.
        x_grid: Spatial coordinates [N].
        t_grid: Temporal coordinates [M].

    Returns:
        u_pred: Solution values [N, M].
    """
    device = next(model.parameters()).device
    xx, tt = np.meshgrid(x_grid, t_grid, indexing="ij")
    points = np.stack([xx.ravel(), tt.ravel()], axis=1)
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device)

    u_pred = model(points_tensor).cpu().numpy().reshape(xx.shape)
    return u_pred


def export_checkpoint_summary(checkpoint_path: str, output_path: str = None):
    """Export a human-readable summary of a checkpoint."""
    _, metadata = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

    summary = {
        "checkpoint_path": checkpoint_path,
        **metadata,
    }

    if output_path is None:
        output_path = checkpoint_path.replace(".pt", "_summary.json")

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Checkpoint summary saved to: {output_path}")
    print(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    parser = argparse.ArgumentParser(description="NysAdam Model Inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt checkpoint file")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input image (for classification models)")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate on test dataset")
    parser.add_argument("--pinn_grid", type=int, default=None,
                        help="Grid resolution for PINN prediction (e.g., 100)")
    parser.add_argument("--summary", action="store_true",
                        help="Print checkpoint summary and exit")
    parser.add_argument("--export_onnx", type=str, default=None,
                        help="Export model to ONNX format at given path")
    args = parser.parse_args()

    # Summary mode
    if args.summary:
        export_checkpoint_summary(args.checkpoint)
        return

    # Load model
    device = get_device()
    model, metadata = load_checkpoint(args.checkpoint, device)

    print(f"Loaded: {metadata['model_class']} from {args.checkpoint}")
    print(f"  Optimizer: {metadata['optimizer']}")
    print(f"  Dataset: {metadata['dataset']}")
    print(f"  Test Loss: {metadata['test_loss']}")
    print(f"  Test Accuracy: {metadata['test_accuracy']}")
    print()

    # ONNX export
    if args.export_onnx:
        model_class = metadata["model_class"]
        if model_class in ("LogisticRegression", "ResNet"):
            dummy = torch.randn(1, 3, 32, 32, device=device)
        elif model_class == "PINN":
            dummy = torch.randn(1, 2, device=device)
        elif model_class == "TransformerLM":
            dummy = torch.randint(0, 100, (1, 64), device=device)
        else:
            dummy = torch.randn(1, 64, device=device)

        torch.onnx.export(model, dummy, args.export_onnx, verbose=False)
        print(f"Exported ONNX model to: {args.export_onnx}")
        return

    # PINN prediction
    if args.pinn_grid is not None:
        n = args.pinn_grid
        x = np.linspace(-1, 1, n)
        t = np.linspace(0, 1, n)
        u = predict_pinn(model, x, t)
        print(f"PINN prediction on {n}×{n} grid:")
        print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
        print(f"  u shape: {u.shape}")

        # Save prediction
        out_path = args.checkpoint.replace(".pt", "_pinn_prediction.npz")
        np.savez(out_path, x=x, t=t, u=u)
        print(f"  Saved to: {out_path}")
        return

    # Image classification
    if args.input is not None:
        try:
            from PIL import Image
            import torchvision.transforms as T

            transform = T.Compose([
                T.Resize((32, 32)),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            img = Image.open(args.input).convert("RGB")
            tensor = transform(img)

            cifar10_classes = [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck",
            ]

            result = predict_image(model, tensor, class_names=cifar10_classes)
            print(f"Prediction for: {args.input}")
            print(f"  Class: {result.get('class_name', result['class_id'])}")
            print(f"  Confidence: {result['confidence']:.4f}")
            print(f"  Top-5:")
            for r in result["top5"]:
                print(f"    {r['class_name']:>12s}: {r['confidence']:.4f}")
        except ImportError:
            print("PIL not installed. Run: pip install Pillow")
        return

    # Evaluate mode
    if args.evaluate:
        from nysadam.data.datasets import load_cifar, load_sklearn_dataset
        from nysadam.utils.metrics import evaluate

        dataset = metadata.get("dataset", "")
        if "cifar10" in dataset and "cifar100" not in dataset:
            _, _, test_loader, _ = load_cifar("cifar10")
        elif "cifar100" in dataset:
            _, _, test_loader, _ = load_cifar("cifar100")
        elif "digits" in dataset:
            _, _, test_loader, _ = load_sklearn_dataset("digits")
        else:
            print(f"Cannot auto-load test data for dataset: {dataset}")
            return

        results = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        print(f"Evaluation on held-out TEST set (unbiased):")
        print(f"  Loss:      {results['loss']:.4f}")
        if "accuracy" in results:
            print(f"  Accuracy:  {results['accuracy']:.4f}")
        if "f1_macro" in results:
            print(f"  F1-Macro:  {results['f1_macro']:.4f}")
        if "ece" in results:
            print(f"  ECE:       {results['ece']:.4f}")
        return

    print("No action specified. Use --input, --evaluate, --pinn_grid, --summary, or --export_onnx")


if __name__ == "__main__":
    main()
