"""Unit tests for FNO 1D components: GRF generator, spectral solver with theta = c * T, and FNO architecture."""

import unittest
import torch
from src.data_gen import GaussianRandomField1D, solve_heat_1d_spectral
from src.model import SpectralConv1d, FNO1d
from src.loss import RelativeL2Loss


class TestFNO1D(unittest.TestCase):
    def test_grf_sampling(self):
        grf = GaussianRandomField1D(length=1.0, spatial_res=128, tau=5.0, d=2.0)
        u0 = grf.sample(num_samples=8, seed=42)
        self.assertEqual(u0.shape, (8, 128))
        self.assertTrue(torch.all(torch.isfinite(u0)))

    def test_spectral_heat_solver_scaling_invariance(self):
        spatial_res = 128
        u0 = torch.randn(4, spatial_res)

        # At theta = 0, solution matches initial condition
        theta_zero = torch.zeros(4, 1)
        u_t0 = solve_heat_1d_spectral(u0, theta=theta_zero)
        self.assertTrue(torch.allclose(u0, u_t0, atol=1e-5))

        # Scaling invariance: (c1=0.01, t1=1.0) must produce identical output to (c2=0.04, t2=0.25)
        u_sol1 = solve_heat_1d_spectral(u0, c=0.01, t=1.0)
        u_sol2 = solve_heat_1d_spectral(u0, c=0.04, t=0.25)
        self.assertTrue(torch.allclose(u_sol1, u_sol2, atol=1e-6))

        # Higher diffusion product must dissipate more energy
        theta_small = torch.full((4, 1), 0.001)
        theta_large = torch.full((4, 1), 0.02)
        u_small = solve_heat_1d_spectral(u0, theta=theta_small)
        u_large = solve_heat_1d_spectral(u0, theta=theta_large)
        norm_initial = torch.norm(u0, dim=-1)
        norm_small = torch.norm(u_small, dim=-1)
        norm_large = torch.norm(u_large, dim=-1)
        self.assertTrue(torch.all(norm_small <= norm_initial + 1e-5))
        self.assertTrue(torch.all(norm_large <= norm_small + 1e-5))

    def test_spectral_conv_shapes(self):
        conv = SpectralConv1d(in_channels=16, out_channels=16, modes=8)
        # Check standard resolution
        x1 = torch.randn(2, 16, 128)
        out1 = conv(x1)
        self.assertEqual(out1.shape, (2, 16, 128))

        # Check mesh-independence (different spatial resolution N=256)
        x2 = torch.randn(2, 16, 256)
        out2 = conv(x2)
        self.assertEqual(out2.shape, (2, 16, 256))

    def test_fno1d_forward_and_backward(self):
        batch_size = 4
        spatial_res = 128
        model = FNO1d(modes=16, width=32, in_channels=3, out_channels=1, num_layers=4)

        # Input: (batch, spatial_res, 3) representing [u0, x, theta = c * T]
        inputs = torch.randn(batch_size, spatial_res, 3)
        targets = torch.randn(batch_size, spatial_res, 1)

        outputs = model(inputs)
        self.assertEqual(outputs.shape, (batch_size, spatial_res, 1))

        # Test loss computation and backward gradient flow
        criterion = RelativeL2Loss()
        loss = criterion(outputs, targets)
        self.assertTrue(torch.isfinite(loss))

        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Gradient missing for {name}")

    def test_zero_shot_resolution_flexibility(self):
        """Verifies that the same model accepts inputs of arbitrary spatial resolutions."""
        model = FNO1d(modes=16, width=32, in_channels=3, out_channels=1, num_layers=4)
        model.eval()

        for res in [64, 128, 256, 512]:
            inputs = torch.randn(2, res, 3)
            with torch.no_grad():
                out = model(inputs)
            self.assertEqual(out.shape, (2, res, 1))


if __name__ == "__main__":
    unittest.main()
