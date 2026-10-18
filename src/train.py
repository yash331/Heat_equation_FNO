"""Training script for 1D Heat Equation Fourier Neural Operator.

Conditioned on the physical diffusion product: theta = c * T.
"""

import os
import argparse
import time
import yaml
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.model import FNO1d
from src.loss import RelativeL2Loss
from src.dataset import get_dataloaders
from src.data_gen import generate_heat_1d_dataset
from src.utils import get_device, save_checkpoint, plot_training_curves, plot_predictions


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Runs one training epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)    # (batch, spatial_res, 3)
        targets = targets.to(device)  # (batch, spatial_res, 1)

        optimizer.zero_grad()
        predictions = model(inputs)   # (batch, spatial_res, 1)

        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluates model on validation or test dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for inputs, targets, _ in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        loss = criterion(predictions, targets)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def ensure_datasets_exist(data_cfg):
    """Checks for dataset files and generates them if missing."""
    save_dir = data_cfg.get("save_dir", "data")
    os.makedirs(save_dir, exist_ok=True)
    train_path = os.path.join(save_dir, "train_data.pt")
    val_path = os.path.join(save_dir, "val_data.pt")
    test_path = os.path.join(save_dir, "test_data.pt")

    theta_min = data_cfg.get("theta_min", 0.0001)
    theta_max = data_cfg.get("theta_max", 0.05)

    if not (os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path)):
        print(f"Dataset files not found. Auto-generating datasets (theta = c * T in [{theta_min}, {theta_max}])...")
        train_data = generate_heat_1d_dataset(
            num_samples=data_cfg["num_samples_train"],
            spatial_res=data_cfg["spatial_res"],
            length=data_cfg["length"],
            theta_min=theta_min,
            theta_max=theta_max,
            tau=data_cfg["grf_tau"],
            d=data_cfg["grf_d"],
            seed=123,
        )
        torch.save(train_data, train_path)

        val_data = generate_heat_1d_dataset(
            num_samples=data_cfg["num_samples_val"],
            spatial_res=data_cfg["spatial_res"],
            length=data_cfg["length"],
            theta_min=theta_min,
            theta_max=theta_max,
            tau=data_cfg["grf_tau"],
            d=data_cfg["grf_d"],
            seed=456,
        )
        torch.save(val_data, val_path)

        test_data = generate_heat_1d_dataset(
            num_samples=data_cfg["num_samples_test"],
            spatial_res=data_cfg["spatial_res"],
            length=data_cfg["length"],
            theta_min=theta_min,
            theta_max=theta_max,
            tau=data_cfg["grf_tau"],
            d=data_cfg["grf_d"],
            seed=789,
        )
        torch.save(test_data, test_path)
        print(f"Data generation complete. Saved to '{save_dir}/'.")

    return train_path, val_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Train 1D Heat Equation FNO with theta = c * T")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', 'mps', 'auto')")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("eval", {})

    # Overrides from CLI if specified
    epochs = args.epochs if args.epochs is not None else train_cfg["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else train_cfg["batch_size"]
    lr = args.lr if args.lr is not None else train_cfg["learning_rate"]
    device_pref = args.device if args.device is not None else train_cfg["device"]

    device = get_device(device_pref)
    print(f"Using device: {device}")

    # Prepare datasets and dataloaders
    train_path, val_path, test_path = ensure_datasets_exist(data_cfg)
    train_loader, val_loader, test_loader = get_dataloaders(
        train_path, val_path, test_path, batch_size=batch_size
    )

    # Initialize model
    model = FNO1d(
        modes=model_cfg["modes"],
        width=model_cfg["width"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        num_layers=model_cfg["num_layers"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized FNO-1D model with {total_params:,} trainable parameters.")

    # Setup loss, optimizer and scheduler
    criterion = RelativeL2Loss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=train_cfg["weight_decay"])
    scheduler = StepLR(optimizer, step_size=train_cfg["step_size"], gamma=train_cfg["gamma"])

    checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
    results_dir = eval_cfg.get("plot_dir", "results")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    print("Starting FNO training...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(
                f"Epoch [{epoch:3d}/{epochs:3d}] | "
                f"Train Rel L2: {train_loss:.4e} | "
                f"Val Rel L2: {val_loss:.4e} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.2f}s"
                f"{' *BEST*' if is_best else ''}"
            )

        # Checkpoint saving
        save_checkpoint(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            },
            checkpoint_dir=checkpoint_dir,
            is_best=is_best,
        )

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.1f}s. Best Val Rel L2: {best_val_loss:.4e}")

    # Plot and save training curves
    curves_path = os.path.join(results_dir, "training_curves.png")
    plot_training_curves(train_losses, val_losses, save_path=curves_path)

    # Final evaluation on test set with best model
    best_ckpt = os.path.join(checkpoint_dir, "best_model.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print("Loaded best model weights for test evaluation.")

    if test_loader is not None:
        test_loss = evaluate(model, test_loader, criterion, device)
        print(f"Final Test Set Relative L2 Loss: {test_loss:.4e}")

        # Visualize sample predictions
        inputs, targets, theta_vals = next(iter(test_loader))
        inputs, targets = inputs.to(device), targets.to(device)
        model.eval()
        with torch.no_grad():
            preds = model(inputs)

        u0 = inputs[:, :, 0]
        x_grid = inputs[0, :, 1]
        u_true = targets[:, :, 0]
        u_pred = preds[:, :, 0]

        pred_plot_path = os.path.join(results_dir, "test_predictions.png")
        plot_predictions(
            u0, u_true, u_pred, x_grid, theta_vals, save_path=pred_plot_path, num_samples=3
        )


if __name__ == "__main__":
    main()
