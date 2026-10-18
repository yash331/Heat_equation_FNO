"""Utility functions for FNO: visualization, device management, and checkpoints."""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_device(device_preference: str = "auto") -> torch.device:
    """Selects compute device based on preference and hardware availability."""
    if device_preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_preference)
    return device


def plot_predictions(u0, u_true, u_pred, x, theta, save_path=None, num_samples=3):
    """Plots initial condition, ground truth solution, FNO prediction, and error.

    Args:
        u0: (num_samples, spatial_res)
        u_true: (num_samples, spatial_res)
        u_pred: (num_samples, spatial_res)
        x: (spatial_res,)
        theta: (num_samples,) [effective diffusion product theta = c * T]
        save_path: Optional path to save figure.
        num_samples: Number of sample profiles to display.
    """
    if isinstance(u0, torch.Tensor):
        u0 = u0.detach().cpu().numpy()
    if isinstance(u_true, torch.Tensor):
        u_true = u_true.detach().cpu().numpy()
    if isinstance(u_pred, torch.Tensor):
        u_pred = u_pred.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(theta, torch.Tensor):
        theta = theta.detach().cpu().numpy().flatten()

    num_samples = min(num_samples, len(u0))
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3.2 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(num_samples):
        theta_val = theta[i] if i < len(theta) else 0.0
        # Profile comparison
        ax1 = axes[i, 0]
        ax1.plot(x, u0[i], "k--", label=r"Initial $u_0(x)$", alpha=0.7)
        ax1.plot(x, u_true[i], "b-", label=rf"True $u(x, \theta={theta_val:.4f})$", linewidth=2.0)
        ax1.plot(x, u_pred[i], "r:", label=r"FNO Prediction $\hat{u}(x, \theta)$", linewidth=2.0)
        ax1.set_title(rf"Sample {i + 1}: Diffusion at $\theta = c \cdot T = {theta_val:.4f}$")
        ax1.set_xlabel("x")
        ax1.set_ylabel("Temperature u")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Pointwise absolute error
        ax2 = axes[i, 1]
        err = np.abs(u_true[i] - u_pred[i])
        ax2.plot(x, err, "m-", label=r"Pointwise Absolute Error $|u - \hat{u}|$")
        ax2.set_title(f"Sample {i + 1} Error (Max = {err.max():.2e})")
        ax2.set_xlabel("x")
        ax2.set_ylabel("Absolute Error")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Prediction plot saved to {save_path}")
    plt.close()


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plots training and validation relative L2 loss curves."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "b-", label="Train Relative L2 Loss", linewidth=1.5)
    plt.plot(epochs, val_losses, "r--", label="Validation Relative L2 Loss", linewidth=1.5)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Relative L2 Error (log scale)")
    plt.title("FNO Training and Validation Convergence")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Loss curves saved to {save_path}")
    plt.close()


def plot_resolution_invariance(resolutions, errors, save_path=None):
    """Plots Relative L2 error across varying spatial resolutions."""
    plt.figure(figsize=(8, 5))
    plt.plot(resolutions, errors, "o-", color="teal", linewidth=2.0, markersize=7)
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.xticks(resolutions, labels=[str(r) for r in resolutions])
    plt.xlabel("Spatial Resolution (N points)")
    plt.ylabel("Relative L2 Error (log scale)")
    plt.title("Zero-Shot Super-Resolution (Mesh-Independence Benchmark)")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Mesh invariance plot saved to {save_path}")
    plt.close()


def save_checkpoint(state, checkpoint_dir, is_best=False, filename="checkpoint.pth"):
    """Saves model and optimizer state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_filepath)
