"""
Utility functions for Gymnasium + MuJoCo experiments.
"""

import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def create_env(
    env_id: str = "HalfCheetah-v5",
    render_mode: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    """
    Create a Gymnasium MuJoCo environment with common configurations.
    
    Args:
        env_id: The environment ID (e.g., "HalfCheetah-v5", "Ant-v5")
        render_mode: Rendering mode ("human", "rgb_array", or None)
        max_episode_steps: Maximum steps per episode (None for default)
    
    Returns:
        A configured Gymnasium environment
    """
    env = gym.make(
        env_id,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    return env


def record_video(
    env_id: str,
    policy_fn,
    output_path: str,
    num_episodes: int = 1,
    fps: int = 30,
) -> None:
    """
    Record a video of an agent interacting with an environment.
    
    Args:
        env_id: The environment ID
        policy_fn: A function that takes observation and returns action
        output_path: Path to save the video
        num_episodes: Number of episodes to record
        fps: Frames per second for the video
    """
    import imageio
    
    env = gym.make(env_id, render_mode="rgb_array")
    frames = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        
        while not done:
            frames.append(env.render())
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    
    env.close()
    
    # Save video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"Video saved to {output_path}")


def random_policy(observation: np.ndarray, action_space: gym.spaces.Space) -> np.ndarray:
    """
    A simple random policy for testing.
    
    Args:
        observation: The current observation (unused)
        action_space: The action space to sample from
    
    Returns:
        A random action
    """
    return action_space.sample()


def print_env_info(env: gym.Env) -> None:
    """
    Print detailed information about an environment.
    
    Args:
        env: A Gymnasium environment
    """
    print("=" * 60)
    print(f"Environment: {env.spec.id if env.spec else 'Unknown'}")
    print("=" * 60)
    print(f"\nObservation Space:")
    print(f"  Type: {type(env.observation_space).__name__}")
    print(f"  Shape: {env.observation_space.shape}")
    if hasattr(env.observation_space, 'low'):
        print(f"  Low: {env.observation_space.low[:5]}...")
        print(f"  High: {env.observation_space.high[:5]}...")
    
    print(f"\nAction Space:")
    print(f"  Type: {type(env.action_space).__name__}")
    print(f"  Shape: {env.action_space.shape}")
    if hasattr(env.action_space, 'low'):
        print(f"  Low: {env.action_space.low}")
        print(f"  High: {env.action_space.high}")
    
    print(f"\nMax Episode Steps: {env.spec.max_episode_steps if env.spec else 'Unknown'}")
    print("=" * 60)
