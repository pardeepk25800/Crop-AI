# evaluate.py — Model Evaluation & Reporting for CropAI
# Confusion matrix, classification report, GradCAM, and metrics comparison.

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional

from config import RESULTS_DIR, DISEASE_CLASSES

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  CLASSIFICATION METRICS (Disease Detection)
# ─────────────────────────────────────────────────────────────────────────────

def generate_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                    class_names: List[str] = None,
                                    save_path: str = None) -> dict:
    """
    Generate a comprehensive classification report with per-class metrics.

    Args:
        y_true:      Ground truth labels (int array)
        y_pred:      Predicted labels (int array)
        class_names: Optional list of class name strings
        save_path:   Optional path to save the report as JSON

    Returns:
        dict with 'accuracy', 'macro_f1', 'weighted_f1', and per-class metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        classification_report as sklearn_report
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(precision))]

    report = {
        "accuracy":    round(accuracy, 4),
        "macro_f1":    round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision":    round(macro_p, 4),
        "macro_recall":       round(macro_r, 4),
        "per_class": {
            name: {
                "precision": round(float(p), 4),
                "recall":    round(float(r), 4),
                "f1_score":  round(float(f), 4),
                "support":   int(s),
            }
            for name, p, r, f, s in zip(class_names, precision, recall, f1, support)
        }
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[Evaluate] ✓ Classification report saved → {save_path}")

    # Print summary
    print(f"\n{'─' * 50}")
    print(f"  Classification Report")
    print(f"{'─' * 50}")
    print(f"  Accuracy       : {accuracy * 100:.2f}%")
    print(f"  Macro F1       : {macro_f1 * 100:.2f}%")
    print(f"  Weighted F1    : {weighted_f1 * 100:.2f}%")
    print(f"  Macro Precision: {macro_p * 100:.2f}%")
    print(f"  Macro Recall   : {macro_r * 100:.2f}%")

    return report


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: List[str] = None,
                          normalize: bool = True,
                          save_path: str = None,
                          figsize: tuple = (16, 14)):
    """
    Generate and save a beautiful confusion matrix heatmap.

    Args:
        y_true:      Ground truth labels
        y_pred:      Predicted labels
        class_names: Human-readable class names
        normalize:   Whether to normalize values (show percentages)
        save_path:   Where to save the plot
        figsize:     Figure size tuple
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_display = cm_normalized
        fmt = ".1%"
        title = "Normalized Confusion Matrix"
    else:
        cm_display = cm
        fmt = "d"
        title = "Confusion Matrix"

    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]

    # Shorten class names for display
    short_names = [name.split("___")[-1][:20] if "___" in name else name[:20]
                   for name in class_names]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display, annot=True, fmt=fmt, cmap="Greens",
        xticklabels=short_names, yticklabels=short_names,
        square=True, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Proportion" if normalize else "Count"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "confusion_matrix.png")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] ✓ Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  REGRESSION METRICS (Yield Prediction)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray,
                        label: str = "Test") -> dict:
    """
    Evaluate regression model performance with multiple metrics.

    Returns:
        dict with MAE, RMSE, R², MAPE, and explained variance
    """
    from sklearn.metrics import (
        mean_absolute_error, mean_squared_error,
        r2_score, mean_absolute_percentage_error,
        explained_variance_score
    )

    metrics = {
        "mae":                round(mean_absolute_error(y_true, y_pred), 2),
        "rmse":               round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "r2":                 round(r2_score(y_true, y_pred), 4),
        "mape":               round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
        "explained_variance": round(explained_variance_score(y_true, y_pred), 4),
    }

    print(f"\n{'─' * 40}")
    print(f"  {label} Regression Metrics")
    print(f"{'─' * 40}")
    print(f"  MAE  : {metrics['mae']:>10.2f} kg/ha")
    print(f"  RMSE : {metrics['rmse']:>10.2f} kg/ha")
    print(f"  R²   : {metrics['r2']:>10.4f}")
    print(f"  MAPE : {metrics['mape']:>10.2f}%")
    print(f"  EV   : {metrics['explained_variance']:>10.4f}")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 3.  GRAD-CAM VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_gradcam(model, image_tensor, target_layer=None,
                     class_idx: int = None, save_path: str = None):
    """
    Generate Grad-CAM heatmap for a given image and model.
    Shows which regions of the leaf image the CNN focused on.

    Args:
        model:        PyTorch CNN model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        target_layer: The layer to compute gradients for
        class_idx:    Target class (if None, uses predicted class)
        save_path:    Where to save the Grad-CAM overlay

    Returns:
        heatmap (numpy array), overlay (numpy array)
    """
    import torch
    import torch.nn.functional as F

    model.eval()

    # Hook storage
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    # Register hooks on target layer
    if target_layer is None:
        # Default: last convolutional layer of backbone
        children = list(model.backbone.children())
        target_layer = children[-2] if len(children) > 2 else children[-1]

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image_tensor)

    if class_idx is None:
        class_idx = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1.0
    output.backward(gradient=one_hot)

    # Compute Grad-CAM
    act = activations[0][0]   # (C, H, W)
    grad = gradients[0][0]    # (C, H, W)

    weights = grad.mean(dim=(1, 2))  # Global average pooling of gradients
    cam = (weights[:, None, None] * act).sum(dim=0)
    cam = F.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.cpu().numpy()

    # Resize to original image size
    from PIL import Image as PILImage
    cam_resized = np.array(PILImage.fromarray(
        (cam * 255).astype(np.uint8)
    ).resize((image_tensor.shape[3], image_tensor.shape[2]))) / 255.0

    # Cleanup hooks
    fh.remove()
    bh.remove()

    # Generate overlay
    if save_path:
        _save_gradcam_overlay(image_tensor, cam_resized, save_path)

    return cam_resized


def _save_gradcam_overlay(image_tensor, heatmap, save_path):
    """Save Grad-CAM overlay combining original image + heatmap."""
    import torch

    # Denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor[0].cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)

    # Create heatmap overlay
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(img)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.suptitle("Grad-CAM — Model Attention Visualization", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] ✓ Grad-CAM overlay saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MODEL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def compare_models(results: dict, save_path: str = None):
    """
    Generate a comparison table / chart for multiple model results.

    Args:
        results: Dict of model_name → {metric: value} dicts
        save_path: Optional save path for the chart

    Example:
        compare_models({
            "XGBoost":       {"r2": 0.91, "mae": 180},
            "RandomForest":  {"r2": 0.88, "mae": 210},
            "Ensemble":      {"r2": 0.93, "mae": 165},
        })
    """
    df = pd.DataFrame(results).T

    print(f"\n{'─' * 50}")
    print(f"  Model Comparison")
    print(f"{'─' * 50}")
    print(df.to_string())

    # Bar chart
    fig, axes = plt.subplots(1, len(df.columns), figsize=(5 * len(df.columns), 4))
    if len(df.columns) == 1:
        axes = [axes]

    colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(df)))

    for ax, col in zip(axes, df.columns):
        bars = ax.bar(df.index, df[col], color=colors, edgecolor="white")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel(col)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] ✓ Model comparison chart saved → {save_path}")

    return df
