#!/usr/bin/env python
"""
Train a PPO agent on a MuJoCo environment using Stable-Baselines3.

This script demonstrates:
1. Creating a vectorized environment
2. Training with PPO algorithm
3. Saving and loading models
4. Logging with TensorBoard
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import gymnasium as gym

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize
except ImportError:
    print("Error: stable-baselines3 is required for this example.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)


def get_paths(env_id: str, run_name: str = None) -> dict:
    """Get paths for saving models and logs."""
    project_root = Path(__file__).parent.parent
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{env_id}_{timestamp}"
    
    return {
        "models": project_root / "models" / run_name,
        "logs": project_root / "logs" / run_name,
        "best_model": project_root / "models" / run_name / "best_model",
    }


def train(
    env_id: str = "HalfCheetah-v5",
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    run_name: str = None,
    eval_freq: int = 10_000,
    save_freq: int = 25_000,
) -> None:
    """
    Train a PPO agent on the specified environment.
    
    Args:
        env_id: Gymnasium environment ID
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate for PPO
        run_name: Name for this training run
        eval_freq: Evaluation frequency (in timesteps)
        save_freq: Checkpoint save frequency (in timesteps)
    """
    paths = get_paths(env_id, run_name)
    
    # Create directories
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    
    print("=" * 60)
    print(f"Training PPO on {env_id}")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Model path: {paths['models']}")
    print(f"Log path: {paths['logs']}")
    print("=" * 60)
    
    # Create vectorized training environment
    print("\nCreating environments...")
    train_env = make_vec_env(env_id, n_envs=n_envs)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = make_vec_env(env_id, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["best_model"]),
        log_path=str(paths["logs"]),
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=5,
        deterministic=True,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // n_envs, 1),
        save_path=str(paths["models"]),
        name_prefix="ppo_checkpoint",
    )
    
    # Create PPO model
    print("Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=str(paths["logs"]),
    )
    
    # Train
    print("\nStarting training...")
    print("(Monitor with TensorBoard: tensorboard --logdir logs)")
    print("-" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = paths["models"] / "final_model"
    model.save(str(final_model_path))
    train_env.save(str(paths["models"] / "vec_normalize.pkl"))
    
    print("-" * 60)
    print(f"\nTraining complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {paths['best_model']}")
    print(f"\nTo evaluate: python examples/evaluate_agent.py --model {final_model_path}")
    
    train_env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO on MuJoCo environment")
    parser.add_argument(
        "--env",
        type=str,
        default="HalfCheetah-v5",
        help="Environment ID (default: HalfCheetah-v5)"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100000)"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments (default: 4)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Run name (default: auto-generated)"
    )
    args = parser.parse_args()
    
    train(
        env_id=args.env,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        learning_rate=args.lr,
        run_name=args.name,
    )


if __name__ == "__main__":
    main()
