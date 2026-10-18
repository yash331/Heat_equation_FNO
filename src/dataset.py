"""PyTorch Dataset and DataLoader for 1D Heat Equation FNO training.

Conditioned on the effective diffusion product:
    theta = c * T  (where c is diffusion constant and T is time)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader


class Heat1DDataset(Dataset):
    """Dataset for 1D Heat Equation parameterized by theta = c * T.

    Input features per spatial point x:
        - Channel 0: Initial condition u_0(x)
        - Channel 1: Spatial coordinate x in [0, L]
        - Channel 2: Effective diffusion product theta = c * T (broadcast uniformly)
    Target:
        - Diffused temperature field u(x, T)
    """

    def __init__(self, data_source):
        """
        Args:
            data_source: File path (.pt) or dict containing 'u0', 'theta' (or 't'), 'u_target', 'x'.
        """
        if isinstance(data_source, str):
            if not os.path.exists(data_source):
                raise FileNotFoundError(f"Data file not found at: {data_source}")
            data = torch.load(data_source, map_location="cpu")
        elif isinstance(data_source, dict):
            data = data_source
        else:
            raise TypeError("data_source must be a file path (str) or a dictionary")

        self.u0 = data["u0"].float()              # (N_samples, spatial_res)
        # Support either 'theta' or 't' (if legacy dataset)
        if "theta" in data:
            self.theta = data["theta"].float()    # (N_samples, 1)
        elif "t" in data:
            # If alpha was provided in legacy data, compute theta = alpha * t
            alpha = data.get("alpha", 0.01)
            self.theta = (data["t"] * alpha).float()
        else:
            raise KeyError("Dataset must contain 'theta' or 't'")

        self.u_target = data["u_target"].float()  # (N_samples, spatial_res)
        self.x = data["x"].float()                # (spatial_res,)
        self.spatial_res = self.u0.shape[-1]

    def __len__(self):
        return self.u0.shape[0]

    def __getitem__(self, idx):
        u0_i = self.u0[idx]                # (spatial_res,)
        theta_i = self.theta[idx]          # (1,)
        target_i = self.u_target[idx]      # (spatial_res,)
        x_grid = self.x                    # (spatial_res,)

        # Broadcast effective diffusion product theta across all spatial locations
        theta_grid = theta_i.repeat(self.spatial_res)  # (spatial_res,)

        # Stack into 3-channel input tensor: [u_0(x), x, theta = c * T]
        inputs = torch.stack([u0_i, x_grid, theta_grid], dim=-1)

        # Target shape: (spatial_res, 1)
        target = target_i.unsqueeze(-1)

        return inputs, target, theta_i


def get_dataloaders(train_path, val_path, test_path=None, batch_size=32, num_workers=0):
    """Creates train, val, and optional test DataLoaders."""
    train_dataset = Heat1DDataset(train_path)
    val_dataset = Heat1DDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = None
    if test_path and os.path.exists(test_path):
        test_dataset = Heat1DDataset(test_path)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader, test_loader
