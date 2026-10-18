# 🌊 1D Heat Equation Fourier Neural Operator (FNO)

An implementation of the **Fourier Neural Operator (FNO)** for learning the solution operator of the 1D Heat Diffusion Equation with **flexible diffusion constant $c$**, **arbitrary evaluation time $T$**, and **mesh-independent zero-shot super-resolution**, based on the paper:

> **Fourier Neural Operator for Parametric Partial Differential Equations**  
> *Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, Anima Anandkumar*  
> International Conference on Learning Representations (ICLR), 2021.  
> [arXiv:2010.08895](https://arxiv.org/abs/2010.08895)

---

## 📌 Problem Formulation

The 1D Heat Diffusion Equation is defined on a periodic domain:

$$\frac{\partial u(x,t)}{\partial t} = c \frac{\partial^2 u(x,t)}{\partial x^2}, \quad x \in [0, L], \quad t \ge 0$$

with periodic boundary conditions $u(0, t) = u(L, t)$, where $c > 0$ is the **thermal diffusivity constant** and $t$ is **time**.

### 💡 Physics Scaling Law: The $\theta = c \cdot T$ Invariance
In Fourier space, the exact physical solution is given by:

$$\hat{u}(k, T) = \hat{u}_0(k) \cdot \exp\left(-\left(\frac{2\pi k}{L}\right)^2 (c \cdot T)\right)$$

The physical diffusion field depends **exclusively on the product**:
$$\theta = c \cdot T$$
*(the Fourier number $\text{Fo} = \frac{c T}{L^2}$ for $L=1$)*.

For example:
* Material A ($c = 0.04$, high diffusivity) at short duration $T = 0.25 \implies \theta = 0.01$
* Material B ($c = 0.01$, low diffusivity) at longer duration $T = 1.00 \implies \theta = 0.01$

Both scenarios yield the **exact same physical temperature profile**. By conditioning the neural operator on $\theta = c \cdot T$, the network becomes **universal for any material conductivity $c$ and any time horizon $T$**!

$$\mathcal{G}_\theta : (u_0(x), \, x, \, \theta = c \cdot T) \longmapsto u(x, T)$$

---

## 🚀 Key Innovations & Features

1. **Universal Diffusion Product ($\theta = c \cdot T$) Conditioning**:
   - The effective diffusion scale $\theta = c \cdot T$ is broadcast across space as an input channel: $[u_0(x), x, \theta]$.
   - Allows users to freely specify any pair $(c, T)$ or pass $\theta$ directly.
   - Preserves exact physical similarity invariance $(c_1 \cdot T_1 = c_2 \cdot T_2)$.

2. **Zero Numerical Error Data Generation (Exact Spectral Method)**:
   - Initial conditions are drawn from **Gaussian Random Fields (GRF)**: $u_0 \sim \mathcal{N}\left(0, (-\Delta + \tau^2 I)^{-d}\right)$.
   - Due to periodic boundary conditions, the solution is calculated **analytically in Fourier space**:
     $$\hat{u}(k) = \hat{u}_0(k) \exp\left(-(2\pi k / L)^2 \theta\right)$$
   - Zero discretization error and generation of thousands of samples in seconds!

3. **Zero-Shot Super-Resolution (Mesh Independence)**:
   - Trained on resolution $N = 128$, the exact same model weights can be directly evaluated on $N = 64, 256, 512, 1024$ without retraining.

4. **Modular Architecture & Standalone Notebook**:
   - Production-ready Python codebase (`src/`, `configs/`, `tests/`).
   - 1-click standalone Jupyter notebook in `notebooks/fno_1d_heat_equation.ipynb` ready for **Google Colab** and **Kaggle GPU** execution.

---

## 📂 Project Structure

```text
FNNP/
├── configs/
│   └── config.yaml             # Hyperparameters (modes, width, lr, epochs, theta_min/max)
├── data/
│   └── README.md               # Data storage directory (auto-populated)
├── notebooks/
│   └── fno_1d_heat_equation.ipynb # Ready-to-run standalone notebook for Colab / Kaggle
├── src/
│   ├── __init__.py
│   ├── data_gen.py             # GRF generator & exact Fourier spectral solver with theta = c * T
│   ├── dataset.py              # PyTorch Dataset and DataLoader ([u0, x, theta] channels)
│   ├── model.py                # SpectralConv1d and FNO1d architecture
│   ├── loss.py                 # Relative L2 Loss (LpLoss from paper)
│   ├── train.py                # Training loop with validation & checkpointing
│   ├── evaluate.py             # Test evaluation, mesh-invariance & scaling invariance verification
│   └── utils.py                # Plotting utilities & model checkpoint helpers
├── tests/
│   ├── __init__.py
│   └── test_components.py      # Unit tests (shapes, backprop, mesh invariance, scaling law)
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation and usage guide
```

---

## ⚡ Quickstart Guide

### 1. Installation
Clone or download the repository, then install requirements:
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset (Optional: auto-generated during training)
```bash
python3 -m src.data_gen --config configs/config.yaml
```

### 3. Train the FNO Model
```bash
python3 -m src.train --config configs/config.yaml
```
CLI overrides are supported:
```bash
python3 -m src.train --epochs 100 --batch_size 32 --lr 0.001 --device auto
```

### 4. Evaluate & Verify Invariance
```bash
python3 -m src.evaluate --config configs/config.yaml --checkpoint checkpoints/best_model.pth
```
This generates:
* **Metric summary**: Relative $L_2$, RMSE, and $L_\infty$ maximum error.
* **Scaling Invariance Verification**: Confirms that $(c_1 \cdot T_1) = (c_2 \cdot T_2)$ yields identical outputs.
* `results/mesh_invariance.png`: Relative error across spatial resolutions $N \in [64, 128, 256, 512, 1024]$.
* `results/diffusion_rollout.png`: Continuous diffusion visualization across varying $\theta = c \cdot T$.

### 5. Run Unit Tests
```bash
python3 -m unittest discover -s tests
```

---

## ☁️ Running on Google Colab or Kaggle

1. Open [Google Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/).
2. Select **File -> Upload notebook** and choose:
   `notebooks/fno_1d_heat_equation.ipynb`
3. Switch runtime to GPU:
   * **Google Colab**: *Runtime -> Change runtime type -> T4 GPU*.
   * **Kaggle**: *Settings -> Accelerator -> GPU T4 x2*.
4. Run all cells. It includes an **interactive prediction function** `predict_temperature(model, u0, c=..., T=...)` where you can pass any custom initial condition, diffusion constant $c$, and time $T$!
