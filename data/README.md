# Datasets Directory

This directory stores generated synthetic datasets for the 1D Heat Equation:
- `train_data.pt`: 2,000 training pairs `(u_0, t) -> u(x, t)`
- `val_data.pt`: 400 validation pairs
- `test_data.pt`: 400 test pairs

To generate or re-generate datasets:
```bash
python -m src.data_gen --config configs/config.yaml
```
Datasets are generated via exact analytical Fourier spectral integration with periodic boundary conditions and Gaussian Random Field initial conditions.
