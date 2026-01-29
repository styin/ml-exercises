#!/usr/bin/env python
"""
Demonstration of the custom PointMass environment.
This script verifies that the custom environment works correctly with Gymnasium's API.
"""

import sys
import os
import time
import numpy as np
import gymnasium as gym

# SYSTEM PATH HACK:
# We need to add the project root to sys.path so Python can find the 'src' package.
# In a real deployed package, you wouldn't need this if you installed with 'pip install -e .'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# IMPORTING SRC:
# This is CRITICAL. The __init__.py in src/envs runs the `register()` function.
# Without this import, gym.make("PointMass-v0") would fail with `Error: Environment point_mass-v0 not found`.
import src

def main():
    print("Creating PointMass-v0 environment...")
    
    # 1. MAKE ENVIRONMENT
    # We use render_mode="human" to pop up a window and see the agent moving.
    # The ID "PointMass-v0" matches what we registered in src/envs/__init__.py
    env = gym.make("PointMass-v0", render_mode="human")
    
    # Inspection: Check the spaces
    # Box(6,) means we get 6 float numbers as observation
    print(f"Observation Space: {env.observation_space}")
    # Box(2,) means we must provide 2 float numbers (forces) as action
    print(f"Action Space: {env.action_space}")
    
    # 2. RESET
    # Always reset before the first step.
    # obs: The initial observation (numpy array)
    # info: Dictionary with auxiliary info (often empty)
    obs, info = env.reset()
    print("Environment reset. Running loop...")
    
    # 3. INTERACTION LOOP
    # Run for 500 steps (or until episode ends)
    for i in range(500):
        # Sample a random action from the action space
        # In a real training loop, your Agent/Policy would determine this action.
        action = env.action_space.sample()
        
        # 4. STEP
        # Apply the action to the environment.
        # Returns:
        #   obs: The new state observation
        #   reward: Float value indicating how good the action was
        #   terminated: Bool, true if the agent reached a terminal state (goal)
        #   truncated: Bool, true if time limit reached or other artificial stop
        #   info: Dict, extra debug info
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Print basic telemetry every 50 steps
        if i % 50 == 0:
            agent_pos = obs[:2]   # First 2 values are x,y
            target_pos = obs[4:6] # Last 2 values are target x,y
            print(f"Step {i}: Agent={agent_pos}, Target={target_pos}, Reward={reward:.3f}")
        
        # 5. HANDLE EPISODE END
        if terminated or truncated:
            print("Episode finished!")
            # Basic reset to start over instantly
            obs, info = env.reset()
            
        # Optional: Sleep slightly to make the visualization easier to follow for human eyes
        # (Physics runs much faster than real-time usually)
        time.sleep(0.02)
        
    # 6. CLOSE
    # Clean up resources (close window)
    env.close()
    print("Done!")

if __name__ == "__main__":
    main()
