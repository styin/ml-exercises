#!/usr/bin/env python
"""
Evaluate a trained agent on a MuJoCo environment.

This script demonstrates:
1. Loading a trained model
2. Running evaluation episodes
3. Optionally recording a video
"""

import argparse
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError:
    print("Error: stable-baselines3 is required for this example.")
    print("Install with: pip install stable-baselines3")
    sys.exit(1)


def evaluate(
    model_path: str,
    env_id: str = "HalfCheetah-v5",
    n_episodes: int = 5,
    render: bool = True,
    record_video: bool = False,
    vec_normalize_path: str = None,
) -> None:
    """
    Evaluate a trained agent.
    
    Args:
        model_path: Path to the saved model
        env_id: Gymnasium environment ID
        n_episodes: Number of evaluation episodes
        render: Whether to render the environment
        record_video: Whether to record a video
        vec_normalize_path: Path to VecNormalize stats (optional)
    """
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print(f"Evaluating model: {model_path}")
    print(f"Environment: {env_id}")
    print(f"Episodes: {n_episodes}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = PPO.load(model_path)
    
    # Create environment
    render_mode = "human" if render else ("rgb_array" if record_video else None)
    
    if vec_normalize_path and Path(vec_normalize_path).exists():
        # Use normalized environment if stats are available
        env = DummyVecEnv([lambda: gym.make(env_id, render_mode=render_mode)])
        env = VecNormalize.load(vec_normalize_path, env)
        env.training = False
        env.norm_reward = False
        use_vec_env = True
    else:
        # Use raw environment
        env = gym.make(env_id, render_mode=render_mode)
        use_vec_env = False
    
    # Recording setup
    frames = []
    
    # Run evaluation
    print("\nRunning evaluation...")
    print("-" * 40)
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        if use_vec_env:
            obs = env.reset()
        else:
            obs, info = env.reset()
        
        total_reward = 0.0
        steps = 0
        done = False
        
        while not done:
            # Get action from model
            if use_vec_env:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)
            
            # Take step
            if use_vec_env:
                obs, reward, dones, infos = env.step(action)
                done = dones[0]
                reward = reward[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            # Collect frames for video
            if record_video and not use_vec_env:
                frames.append(env.render())
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}: reward = {total_reward:.2f}, steps = {steps}")
    
    print("-" * 40)
    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Mean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Mean length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min reward: {np.min(episode_rewards):.2f}")
    print(f"  Max reward: {np.max(episode_rewards):.2f}")
    
    # Save video if requested
    if record_video and frames:
        try:
            import imageio
            video_path = project_root / "videos" / f"{env_id}_eval.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(video_path), frames, fps=30)
            print(f"\nVideo saved to: {video_path}")
        except ImportError:
            print("\nWarning: imageio not installed, skipping video recording")
    
    env.close()
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the saved model"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="HalfCheetah-v5",
        help="Environment ID (default: HalfCheetah-v5)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)"
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable rendering"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record evaluation video"
    )
    parser.add_argument(
        "--vec-normalize",
        type=str,
        default=None,
        help="Path to VecNormalize stats file"
    )
    args = parser.parse_args()
    
    evaluate(
        model_path=args.model,
        env_id=args.env,
        n_episodes=args.episodes,
        render=not args.no_render,
        record_video=args.record,
        vec_normalize_path=args.vec_normalize,
    )


if __name__ == "__main__":
    main()
