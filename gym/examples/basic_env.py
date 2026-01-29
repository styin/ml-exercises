#!/usr/bin/env python
"""
Basic example demonstrating Gymnasium MuJoCo environment interaction.

This script shows how to:
1. Create a MuJoCo environment
2. Reset the environment
3. Take random actions
4. Render the environment (optional)
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import print_env_info


def run_episode(env: gym.Env, verbose: bool = True) -> tuple[float, int]:
    """
    Run a single episode with random actions.
    
    Returns:
        Tuple of (total_reward, num_steps)
    """
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    
    while not done:
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        if verbose and steps % 100 == 0:
            print(f"  Step {steps}: reward = {reward:.3f}, total = {total_reward:.3f}")
    
    return total_reward, steps


def main():
    parser = argparse.ArgumentParser(description="Basic Gymnasium MuJoCo environment example")
    parser.add_argument(
        "--env", 
        type=str, 
        default="HalfCheetah-v5",
        help="Environment ID (default: HalfCheetah-v5)"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=3,
        help="Number of episodes to run (default: 3)"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render the environment (opens a window)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Print step-by-step information"
    )
    args = parser.parse_args()
    
    # Create environment
    render_mode = "human" if args.render else None
    env = gym.make(args.env, render_mode=render_mode)
    
    # Print environment information
    print_env_info(env)
    
    # Run episodes
    print(f"\nRunning {args.episodes} episodes with random policy...")
    print("-" * 40)
    
    rewards = []
    steps_list = []
    
    for episode in range(args.episodes):
        total_reward, num_steps = run_episode(env, verbose=args.verbose)
        rewards.append(total_reward)
        steps_list.append(num_steps)
        print(f"Episode {episode + 1}: reward = {total_reward:.2f}, steps = {num_steps}")
    
    print("-" * 40)
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average steps: {np.mean(steps_list):.1f}")
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
