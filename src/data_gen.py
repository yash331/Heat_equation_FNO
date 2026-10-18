"""1D Heat Equation Synthetic Dataset Generator using Exact Spectral Methods.

Generates smooth initial temperature profiles using Gaussian Random Fields (GRF)
under periodic boundary conditions, parameterized by the physical diffusion product:
    theta = c * T  (where c is diffusion constant and T is query time)

Physical equation:
    u_t = c * u_xx,  x in [0, L]
    Solution in Fourier space depends purely on the product theta = c * T:
    u_hat(k, T) = u_0_hat(k) * exp(-(2 * pi * k / L)^2 * (c * T))
"""

import os
import argparse
import yaml
import torch
import numpy as np


class GaussianRandomField1D:
    """Generates 1D periodic Gaussian Random Fields (GRF).

    u_0 ~ N(0, (-Delta + tau^2 * I)^(-d))
    where Delta is the 1D Laplacian with periodic boundary conditions.
    """

    def __init__(self, length=1.0, spatial_res=128, tau=5.0, d=2.0, device="cpu"):
        self.length = length
        self.spatial_res = spatial_res
        self.tau = tau
        self.d = d
        self.device = device

        # Frequencies for real FFT: k = 0, 1, ..., N // 2
        k = torch.arange(0, spatial_res // 2 + 1, device=device, dtype=torch.float32)
        omega = (2.0 * np.pi * k) / length
        self.amplitude = (omega ** 2 + tau ** 2) ** (-d / 2.0)

    def sample(self, num_samples, seed=None):
        """Sample periodic initial conditions.

        Args:
            num_samples: Number of initial profiles to generate.
            seed: Optional random seed.

        Returns:
            torch.Tensor of shape (num_samples, spatial_res).
        """
        if seed is not None:
            torch.manual_seed(seed)

        num_freqs = self.spatial_res // 2 + 1
        real_part = torch.randn(num_samples, num_freqs, device=self.device)
        imag_part = torch.randn(num_samples, num_freqs, device=self.device)

        # DC component imaginary part must be 0 for real signals
        imag_part[:, 0] = 0.0
        if self.spatial_res % 2 == 0:
            imag_part[:, -1] = 0.0

        xi = torch.complex(real_part, imag_part)
        u_hat = xi * self.amplitude.unsqueeze(0)
        u = torch.fft.irfft(u_hat, n=self.spatial_res, dim=-1)

        # Scale by sqrt(N) to normalize energy across resolutions
        return u * np.sqrt(self.spatial_res)


def solve_heat_1d_spectral(u0, theta=None, c=None, t=None, length=1.0):
    """Computes exact analytical solution of 1D heat equation via Fourier spectral method.

    Equation:
        du_hat(k, t)/dt = -c * (2 * pi * k / L)^2 * u_hat(k, t)
    Solution:
        u_hat(k, t) = u_hat_0(k) * exp(-(2 * pi * k / L)^2 * (c * t))

    Args:
        u0: Initial condition tensor of shape (batch, spatial_res)
        theta: Effective diffusion product theta = c * t. Shape (batch, 1) or scalar.
        c: Diffusion constant (optional, if theta is not provided directly).
        t: Time (optional, if theta is not provided directly).
        length: Spatial domain length

    Returns:
        u_t: Solution tensor at effective diffusion theta of shape (batch, spatial_res)
    """
    spatial_res = u0.shape[-1]
    k = torch.arange(0, spatial_res // 2 + 1, device=u0.device, dtype=torch.float32)
    omega = (2.0 * np.pi * k) / length  # (num_freqs,)

    # Determine theta = c * t
    if theta is None:
        if c is None or t is None:
            raise ValueError("Must provide either 'theta' or both 'c' and 't'.")
        theta = c * t

    if isinstance(theta, (int, float)):
        theta = torch.full((u0.shape[0], 1), fill_value=float(theta), device=u0.device, dtype=torch.float32)
    elif isinstance(theta, torch.Tensor) and theta.ndim == 1:
        theta = theta.unsqueeze(-1)

    # Compute Fourier transform of initial condition
    u0_hat = torch.fft.rfft(u0, dim=-1)  # (batch, num_freqs)

    # Decay factor: exp(-omega^2 * theta)
    decay = torch.exp(-(omega ** 2).unsqueeze(0) * theta)

    # Solution in Fourier space
    u_t_hat = u0_hat * decay

    # Back to spatial domain
    u_t = torch.fft.irfft(u_t_hat, n=spatial_res, dim=-1)
    return u_t


def generate_heat_1d_dataset(
    num_samples=2000,
    spatial_res=128,
    length=1.0,
    theta_min=0.0001,
    theta_max=0.05,
    tau=5.0,
    d=2.0,
    seed=42,
    device="cpu",
):
    """Generates complete dataset: u0 (initial profile), theta = c * T, and exact target u(x, T).

    Returns:
        dict with keys:
            'u0': (num_samples, spatial_res)
            'theta': (num_samples, 1)  [effective diffusion product c * T]
            'u_target': (num_samples, spatial_res)
            'x': (spatial_res,)
    """
    grf = GaussianRandomField1D(
        length=length, spatial_res=spatial_res, tau=tau, d=d, device=device
    )

    # Generate initial conditions
    u0 = grf.sample(num_samples, seed=seed)

    # Sample random effective diffusion theta ~ Uniform(theta_min, theta_max)
    if seed is not None:
        torch.manual_seed(seed + 1000)
    theta = torch.rand(num_samples, 1, device=device) * (theta_max - theta_min) + theta_min

    # Compute exact ground truth solutions
    u_target = solve_heat_1d_spectral(u0, theta=theta, length=length)

    # Spatial coordinates grid x in [0, L)
    x = torch.linspace(0, length, spatial_res + 1, device=device)[:-1]

    return {
        "u0": u0.cpu(),
        "theta": theta.cpu(),
        "u_target": u_target.cpu(),
        "x": x.cpu(),
        "length": length,
        "spatial_res": spatial_res,
        "theta_min": theta_min,
        "theta_max": theta_max,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate 1D Heat Equation dataset with theta = c * T")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    save_dir = data_cfg.get("save_dir", "data")
    os.makedirs(save_dir, exist_ok=True)

    theta_min = data_cfg.get("theta_min", 0.0001)
    theta_max = data_cfg.get("theta_max", 0.05)

    print(f"Generating 1D Heat Equation datasets (theta = c * T in [{theta_min}, {theta_max}])...")

    # Generate Train
    print(f"Generating {data_cfg['num_samples_train']} training samples...")
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
    torch.save(train_data, os.path.join(save_dir, "train_data.pt"))

    # Generate Validation
    print(f"Generating {data_cfg['num_samples_val']} validation samples...")
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
    torch.save(val_data, os.path.join(save_dir, "val_data.pt"))

    # Generate Test
    print(f"Generating {data_cfg['num_samples_test']} test samples...")
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
    torch.save(test_data, os.path.join(save_dir, "test_data.pt"))

    print(f"Datasets successfully saved to '{save_dir}/'.")


if __name__ == "__main__":
    main()
