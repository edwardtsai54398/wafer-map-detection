import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, show=False):
    """Plot and return a 2×3 figure of training curves."""
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    pairs = [
        (axes[0, 0], "Loss",      "train_loss",     "val_loss"),
        (axes[0, 1], "F1 Score",  "train_f1",       "val_f1"),
        (axes[0, 2], "Accuracy",  "train_acc",       "val_acc"),
        (axes[1, 0], "Precision", "train_precision", "val_precision"),
        (axes[1, 1], "Recall",    "train_recall",    "val_recall"),
    ]
    for ax, title, train_key, val_key in pairs:
        ax.plot(epochs, [h[train_key] for h in history], marker="o", label="Train")
        ax.plot(epochs, [h[val_key]   for h in history], marker="s", label="Val")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True)

    ax = axes[1, 2]
    ax.plot(epochs, [h["lr"] for h in history], marker="D", color="tab:purple", label="LR")
    ax.set_title("Learning Rate", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(True)

    for row in axes:
        for a in row:
            a.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    fig.suptitle("Training History", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if show:
        plt.show()
    return fig


def plot_cm(cm, num_classes, show=False):
    """Plot and return a row-normalised confusion matrix figure."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xlabel="Predicted Label",
        ylabel="True Label",
        title="Confusion Matrix (Row-Normalised)",
    )

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i,
                f"{cm_norm[i, j]:.2f}\n({cm[i, j]})",
                ha="center", va="center", fontsize=7,
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )

    fig.tight_layout()
    if show:
        plt.show()
    return fig
