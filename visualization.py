# visualization.py — Centralized Plotting & EDA for CropAI
# Training curves, feature importance, data exploration, and summary dashboards.

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Dict, Optional

from config import RESULTS_DIR

os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── Styling ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#FAFAFA",
    "axes.facecolor":    "#FAFAFA",
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "font.family":       "sans-serif",
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

COLORS = {
    "primary":    "#4CAF50",
    "secondary":  "#81C784",
    "accent":     "#FF7043",
    "dark":       "#2E7D32",
    "light":      "#E8F5E9",
    "warning":    "#FFC107",
    "error":      "#F44336",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: str = None,
                          title: str = "Training History"):
    """
    Plot loss and accuracy curves from training history.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        save_path: Where to save the plot
        title: Plot title
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], color=COLORS["primary"],
                 lw=2, label="Train Loss", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"], color=COLORS["accent"],
                 lw=2, label="Val Loss", marker="s", markersize=3)
    axes[0].fill_between(epochs, history["train_loss"],
                          alpha=0.1, color=COLORS["primary"])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].legend(fontsize=9)

    # Accuracy
    train_acc = [a * 100 for a in history["train_acc"]]
    val_acc   = [a * 100 for a in history["val_acc"]]

    axes[1].plot(epochs, train_acc, color=COLORS["primary"],
                 lw=2, label="Train Accuracy", marker="o", markersize=3)
    axes[1].plot(epochs, val_acc, color=COLORS["accent"],
                 lw=2, label="Val Accuracy", marker="s", markersize=3)
    axes[1].fill_between(epochs, train_acc, alpha=0.1, color=COLORS["primary"])

    # Mark best epoch
    best_epoch = np.argmax(val_acc) + 1
    best_acc = max(val_acc)
    axes[1].axvline(best_epoch, color=COLORS["error"], lw=1, linestyle="--",
                     alpha=0.7, label=f"Best: {best_acc:.1f}% (Epoch {best_epoch})")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curve")
    axes[1].legend(fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "training_history.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] ✓ Training curves saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(importances: np.ndarray, feature_names: List[str],
                            top_k: int = 15, save_path: str = None):
    """
    Plot horizontal bar chart of top-K feature importances.
    """
    idx = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in idx]
    values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [COLORS["dark"] if i == 0 else COLORS["primary"]
              for i in range(len(idx))]

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_k} Feature Importances", fontsize=12, fontweight="bold")

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] ✓ Feature importance saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EDA PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """
    Plot correlation matrix heatmap for numeric features.
    """
    numeric_df = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
    sns.heatmap(
        numeric_df.corr(), mask=mask, annot=True, fmt=".2f",
        cmap="RdYlGn", center=0, linewidths=0.5,
        square=True, ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "correlation_heatmap.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] ✓ Correlation heatmap saved → {save_path}")


def plot_distribution(df: pd.DataFrame, columns: List[str] = None,
                      save_path: str = None):
    """
    Plot distribution (histogram + KDE) for specified columns.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns[:6].tolist()

    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten() if n_rows * n_cols > 1 else [axes]

    for i, col in enumerate(columns):
        if i < len(axes):
            ax = axes[i]
            ax.hist(df[col].dropna(), bins=40, color=COLORS["primary"],
                    alpha=0.7, edgecolor="white", density=True)
            df[col].dropna().plot.kde(ax=ax, color=COLORS["dark"], lw=2)
            ax.set_title(col, fontsize=10, fontweight="bold")
            ax.set_ylabel("Density")

    # Hide unused axes
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Feature Distributions", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "feature_distributions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] ✓ Distribution plots saved → {save_path}")


def plot_yield_by_crop(df: pd.DataFrame, save_path: str = None):
    """
    Box plot of yield distribution grouped by crop type.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    crop_order = df.groupby("Crop")["Yield"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="Crop", y="Yield", order=crop_order, palette="Greens_r", ax=ax)

    ax.set_title("Yield Distribution by Crop", fontsize=13, fontweight="bold")
    ax.set_xlabel("Crop Type")
    ax.set_ylabel("Yield (kg/ha)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "yield_by_crop_boxplot.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] ✓ Yield by crop boxplot saved → {save_path}")


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "Actual vs Predicted",
                              save_path: str = None):
    """
    Scatter plot with regression line, R² annotation, and error bands.
    """
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true, y_pred, alpha=0.4, s=12, color=COLORS["primary"],
               edgecolors=COLORS["dark"], linewidth=0.3)

    lim = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], color=COLORS["error"], lw=2, linestyle="--",
            label="Perfect prediction")

    # ±10% error bands
    ax.fill_between([0, lim], [0, lim * 0.9], [0, lim * 1.1],
                     alpha=0.08, color=COLORS["warning"], label="±10% error band")

    ax.set_xlabel("Actual Yield (kg/ha)", fontsize=11)
    ax.set_ylabel("Predicted Yield (kg/ha)", fontsize=11)
    ax.set_title(f"{title}\nR² = {r2:.4f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "actual_vs_predicted.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] ✓ Actual vs predicted saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SUMMARY DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_summary_dashboard(disease_history: dict, yield_metrics: dict,
                                     save_path: str = None):
    """
    Generate a 2×2 summary dashboard with key metrics from both models.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # 1. Disease training loss
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = range(1, len(disease_history["train_loss"]) + 1)
    ax1.plot(epochs, disease_history["train_loss"], color=COLORS["primary"],
             lw=2, label="Train")
    ax1.plot(epochs, disease_history["val_loss"], color=COLORS["accent"],
             lw=2, label="Val")
    ax1.set_title("Disease Model — Loss", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # 2. Disease training accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, [a * 100 for a in disease_history["train_acc"]],
             color=COLORS["primary"], lw=2, label="Train")
    ax2.plot(epochs, [a * 100 for a in disease_history["val_acc"]],
             color=COLORS["accent"], lw=2, label="Val")
    best_acc = max(disease_history["val_acc"]) * 100
    ax2.axhline(best_acc, color=COLORS["error"], lw=1, linestyle="--",
                 alpha=0.5, label=f"Best: {best_acc:.1f}%")
    ax2.set_title("Disease Model — Accuracy", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    # 3. Yield metrics radar (simulated as bar)
    ax3 = fig.add_subplot(gs[1, 0])
    metric_names = list(yield_metrics.keys())
    metric_vals  = list(yield_metrics.values())
    bars = ax3.bar(metric_names, metric_vals, color=[
        COLORS["primary"], COLORS["secondary"],
        COLORS["dark"], COLORS["accent"], COLORS["warning"]
    ][:len(metric_names)], edgecolor="white")
    ax3.set_title("Yield Model — Metrics", fontweight="bold")
    for bar, val in zip(bars, metric_vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. Summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    summary_text = (
        f"CropAI Training Summary\n"
        f"{'─' * 30}\n\n"
        f"🍃 Disease Detection:\n"
        f"   Best Val Accuracy: {best_acc:.2f}%\n"
        f"   Final Train Loss:  {disease_history['train_loss'][-1]:.4f}\n"
        f"   Epochs Trained:    {len(disease_history['train_loss'])}\n\n"
        f"📈 Yield Prediction:\n"
        f"   R² Score:  {yield_metrics.get('r2', 'N/A')}\n"
        f"   MAE:       {yield_metrics.get('mae', 'N/A')} kg/ha\n"
        f"   RMSE:      {yield_metrics.get('rmse', 'N/A')} kg/ha\n"
    )
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor=COLORS["light"], alpha=0.8))

    plt.suptitle("🌾 CropAI — Training Dashboard", fontsize=16,
                 fontweight="bold", y=0.98)

    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "training_dashboard.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] ✓ Training dashboard saved → {save_path}")
