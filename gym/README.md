# Gymnasium + MuJoCo Experiments

A skeleton project for testing and experimenting with [Gymnasium](https://gymnasium.farama.org/) (by Farama Foundation) in conjunction with the [MuJoCo](https://mujoco.org/) physics engine.

## 🚀 Quick Start

### Option A: CPU-Only Setup (No CUDA required)

```bash
# Create the environment
conda env create -f environment-cpu.yml

# Activate
conda activate gym-mujoco-cpu

# Verify installation
python scripts/verify_install.py
```

### Option B: CUDA Setup (Requires NVIDIA GPU)

```bash
# Check your CUDA version first
nvidia-smi

# Create the environment
conda env create -f environment-cuda.yml

# Activate
conda activate gym-mujoco-cuda

# Verify installation
python scripts/verify_install.py
```

### Alternative: Using pip directly

```bash
# CPU-only
pip install -r requirements-cpu.txt

# With CUDA
pip install -r requirements-cuda.txt
```

## ✅ Verify Installation

```bash
python scripts/verify_install.py
```

## 🎯 Run Examples

```bash
# Basic environment test
python examples/basic_env.py

# Train an agent with PPO
python examples/train_ppo.py

# Watch a trained agent
python examples/evaluate_agent.py
```

## 📁 Project Structure

```
gym/
├── environment-cpu.yml    # Conda env (CPU-only)
├── environment-cuda.yml   # Conda env (with CUDA)
├── requirements-cpu.txt   # Pip requirements (CPU-only)
├── requirements-cuda.txt  # Pip requirements (with CUDA)
├── README.md              # This file
├── src/                   # Source code
│   ├── __init__.py
│   └── utils.py           # Utility functions
├── scripts/               # Utility scripts
│   └── verify_install.py  # Installation verification
├── examples/              # Example scripts
│   ├── basic_env.py       # Basic environment interaction
│   ├── train_ppo.py       # PPO training example
│   └── evaluate_agent.py  # Agent evaluation
├── models/                # Saved models (gitignored)
├── logs/                  # Training logs (gitignored)
└── notebooks/             # Jupyter notebooks
    └── exploration.ipynb  # Interactive exploration
```

## 🎮 Available MuJoCo Environments

| Environment | Description | Complexity |
|-------------|-------------|------------|
| `InvertedPendulum-v5` | CartPole with continuous actions | ⭐ |
| `InvertedDoublePendulum-v5` | 2-pole CartPole variant | ⭐⭐ |
| `Reacher-v5` | 2D arm reaching a target | ⭐⭐ |
| `HalfCheetah-v5` | 2D quadruped running | ⭐⭐⭐ |
| `Hopper-v5` | 2D monoped hopping | ⭐⭐⭐ |
| `Walker2d-v5` | 2D biped walking | ⭐⭐⭐ |
| `Swimmer-v5` | 3D robot swimming | ⭐⭐⭐ |
| `Ant-v5` | 3D quadruped running | ⭐⭐⭐⭐ |
| `Humanoid-v5` | 3D humanoid running | ⭐⭐⭐⭐⭐ |
| `HumanoidStandup-v5` | 3D humanoid standing up | ⭐⭐⭐⭐⭐ |

> **Note:** v5 environments are recommended (requires `mujoco>=2.3.3`). They have the most features and fewest bugs.

## 📦 Package Versions

This project uses:
- **Gymnasium**: 1.2.3+ (latest stable)
- **MuJoCo**: 3.4.0+ (latest stable)
- **Stable-Baselines3**: 2.0.0+ (for PPO and other algorithms)
- **PyTorch**: 2.x (CPU or CUDA depending on your setup)

## 📚 References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Farama Foundation](https://farama.org/)

## 📝 License

MIT License
