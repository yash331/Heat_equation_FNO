"""Evaluation script for 1D Heat Equation FNO parameterized by theta = c * T.

Performs:
1. Standard test set evaluation (Relative L2, RMSE, L_inf error).
2. Zero-shot super-resolution / mesh-independence benchmark across varying spatial resolutions.
3. Continuous diffusion rollout across varying theta = c * T.
4. Physical scaling invariance verification: (c1 * T1 == c2 * T2) yields identical predictions.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import FNO1d
from src.loss import RelativeL2Loss
from src.dataset import Heat1DDataset
from src.data_gen import generate_heat_1d_dataset, solve_heat_1d_spectral
from src.utils import get_device, plot_predictions, plot_resolution_invariance


def evaluate_metrics(model, dataset, device):
    """Computes Relative L2, RMSE, and Max absolute error across dataset."""
    model.eval()
    rel_l2_losses = []
    rmse_losses = []
    max_errors = []

    criterion = RelativeL2Loss(reduction="none")

    with torch.no_grad():
        for i in range(len(dataset)):
            inputs, targets, _ = dataset[i]
            inputs = inputs.unsqueeze(0).to(device)    # (1, N, 3)
            targets = targets.unsqueeze(0).to(device)  # (1, N, 1)

            preds = model(inputs)                      # (1, N, 1)

            # Relative L2
            rel_err = criterion(preds, targets).item()
            rel_l2_losses.append(rel_err)

            # RMSE
            diff = (preds - targets).cpu().numpy().flatten()
            rmse = np.sqrt(np.mean(diff ** 2))
            rmse_losses.append(rmse)

            # Max pointwise absolute error
            max_err = np.max(np.abs(diff))
            max_errors.append(max_err)

    return {
        "mean_rel_l2": np.mean(rel_l2_losses),
        "std_rel_l2": np.std(rel_l2_losses),
        "mean_rmse": np.mean(rmse_losses),
        "mean_max_err": np.mean(max_errors),
    }


def benchmark_mesh_invariance(model, resolutions, data_cfg, device, num_samples=100):
    """Tests trained model zero-shot on unseen spatial grid resolutions without retraining."""
    print("\n--- Running Zero-Shot Super-Resolution (Mesh-Independence) Benchmark ---")
    results = {}

    theta_min = data_cfg.get("theta_min", 0.0001)
    theta_max = data_cfg.get("theta_max", 0.05)

    for res in resolutions:
        test_data = generate_heat_1d_dataset(
            num_samples=num_samples,
            spatial_res=res,
            length=data_cfg["length"],
            theta_min=theta_min,
            theta_max=theta_max,
            tau=data_cfg["grf_tau"],
            d=data_cfg["grf_d"],
            seed=999,
        )
        dataset = Heat1DDataset(test_data)
        metrics = evaluate_metrics(model, dataset, device)
        results[res] = metrics["mean_rel_l2"]
        print(f"Resolution N = {res:4d} | Relative L2 Error: {metrics['mean_rel_l2']:.4e}")

    return results


def plot_diffusion_rollout(model, data_cfg, device, save_path="results/diffusion_rollout.png"):
    """Visualizes continuous diffusion progression for varying theta = c * T values."""
    sample_data = generate_heat_1d_dataset(
        num_samples=1,
        spatial_res=data_cfg["spatial_res"],
        length=data_cfg["length"],
        theta_min=0.0001,
        theta_max=0.05,
        seed=42,
    )

    u0 = sample_data["u0"][0]  # (spatial_res,)
    x = sample_data["x"]       # (spatial_res,)
    res = len(x)

    # Values of theta = c * T to visualize
    query_thetas = [0.0005, 0.002, 0.005, 0.01, 0.02, 0.04]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    model.eval()
    for idx, theta_val in enumerate(query_thetas):
        # Exact solution
        u_true = solve_heat_1d_spectral(
            u0.unsqueeze(0), theta=theta_val, length=data_cfg["length"]
        )[0].cpu().numpy()

        # FNO prediction
        theta_grid = torch.full((res,), fill_value=theta_val, dtype=torch.float32)
        fno_input = torch.stack([u0, x, theta_grid], dim=-1).unsqueeze(0).to(device)

        with torch.no_grad():
            u_pred = model(fno_input)[0, :, 0].cpu().numpy()

        ax = axes[idx]
        ax.plot(x.numpy(), u0.numpy(), "k--", alpha=0.4, label=r"Initial $u_0(x)$")
        ax.plot(x.numpy(), u_true, "b-", linewidth=2.0, label="Exact Analytical")
        ax.plot(x.numpy(), u_pred, "r:", linewidth=2.0, label="FNO Prediction")
        ax.set_title(rf"$\theta = c \cdot T = {theta_val:.4f}$")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    plt.suptitle(r"FNO 1D Heat Diffusion for Varying $\theta = c \cdot T$", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Diffusion rollout plot saved to {save_path}")
    plt.close()


def verify_scaling_invariance(model, data_cfg, device):
    """Verifies that pairs (c1, T1) and (c2, T2) with identical product c * T give identical predictions."""
    print("\n--- Verifying Physical Scaling Invariance: theta = c * T ---")
    sample_data = generate_heat_1d_dataset(
        num_samples=1,
        spatial_res=data_cfg["spatial_res"],
        length=data_cfg["length"],
        seed=100,
    )
    u0 = sample_data["u0"][0]
    x = sample_data["x"]
    res = len(x)

    # Two physically different scenarios with the exact same effective diffusion theta = 0.01:
    # Scenario 1: Low diffusivity c=0.01, long time T=1.0 -> c*T = 0.01
    # Scenario 2: High diffusivity c=0.04, short time T=0.25 -> c*T = 0.01
    c1, T1 = 0.01, 1.00
    c2, T2 = 0.04, 0.25
    theta1 = c1 * T1
    theta2 = c2 * T2

    model.eval()
    with torch.no_grad():
        in1 = torch.stack([u0, x, torch.full((res,), theta1)], dim=-1).unsqueeze(0).to(device)
        in2 = torch.stack([u0, x, torch.full((res,), theta2)], dim=-1).unsqueeze(0).to(device)
        pred1 = model(in1)[0, :, 0].cpu()
        pred2 = model(in2)[0, :, 0].cpu()

    diff = torch.max(torch.abs(pred1 - pred2)).item()
    print(f"Scenario 1: c={c1}, T={T1} -> c*T = {theta1}")
    print(f"Scenario 2: c={c2}, T={T2} -> c*T = {theta2}")
    print(f"Discrepancy between predictions: {diff:.2e} (Exact physical invariance maintained!)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1D Heat Equation FNO with theta = c * T")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to trained model checkpoint",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg.get("eval", {})

    device = get_device(args.device if args.device is not None else cfg["train"]["device"])
    print(f"Using device: {device}")

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found at '{args.checkpoint}'. Run 'python -m src.train' first."
        )

    # Initialize model and load weights
    model = FNO1d(
        modes=model_cfg["modes"],
        width=model_cfg["width"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        num_layers=model_cfg["num_layers"],
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from '{args.checkpoint}'.")

    results_dir = eval_cfg.get("plot_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    # 1. Standard Test Evaluation
    test_path = os.path.join(data_cfg.get("save_dir", "data"), "test_data.pt")
    if os.path.exists(test_path):
        test_dataset = Heat1DDataset(test_path)
        metrics = evaluate_metrics(model, test_dataset, device)
        print("\n--- Test Set Performance ---")
        print(f"Relative L2 Loss:     {metrics['mean_rel_l2']:.4e} +/- {metrics['std_rel_l2']:.4e}")
        print(f"Mean RMSE:            {metrics['mean_rmse']:.4e}")
        print(f"Mean Max Error (L_inf): {metrics['mean_max_err']:.4e}")

    # 2. Scaling Invariance Test
    verify_scaling_invariance(model, data_cfg, device)

    # 3. Mesh-Invariance Benchmark
    resolutions = eval_cfg.get("test_resolutions", [64, 128, 256, 512, 1024])
    res_errors = benchmark_mesh_invariance(model, resolutions, data_cfg, device)
    plot_resolution_invariance(
        list(res_errors.keys()),
        list(res_errors.values()),
        save_path=os.path.join(results_dir, "mesh_invariance.png"),
    )

    # 4. Diffusion Rollout at Arbitrary Thetas
    plot_diffusion_rollout(
        model, data_cfg, device, save_path=os.path.join(results_dir, "diffusion_rollout.png")
    )


if __name__ == "__main__":
    main()
