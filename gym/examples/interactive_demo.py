#!/usr/bin/env python
"""
Demonstration of using the native interactive MuJoCo viewer with Gymnasium.
This provides the "solid tabs" UI (Image 2) that allows for detailed inspection,
perturbation, and pausing, which is different from the default Gymnasium viewer.
"""

import sys
import os
import time
import gymnasium as gym
import mujoco
import mujoco.viewer

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src  # Registers the environment

def main():
    print("Creating PointMass-v0 environment...")
    
    # NOTE: We do NOT use render_mode="human" here. 
    # We will handle the viewer manually using mujoco.viewer.
    env = gym.make("PointMass-v0", render_mode=None)
    
    # Reset the environment to initialize the physics
    obs, info = env.reset()
    
    # Extract the underlying MuJoCo model and data objects
    # env.unwrapped gives us the base MujocoEnv
    model = env.unwrapped.model
    data = env.unwrapped.data
    
    print("Launching interactive viewer...")
    print("Toggle the 'Control' panel on the right to see actuators!")
    
    # Launch the passive viewer. 
    # "Passive" means the loop controls the physics, and the viewer just displays it.
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Close the viewer automatically if the window is closed
        while viewer.is_running():
            # 1. Step the Gym environment
            # This advances the physics (data) by one step
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 2. Sync the viewer
            # This tells the viewer "the physics state has changed, please update the graphics"
            viewer.sync()
            
            # 3. Handle resets
            if terminated or truncated:
                env.reset()
                
            # 4. Timing
            # The viewer handles some timing, but we add a sleep to not run too fast
            # In a real training loop, this might be removed or adjusted
            time.sleep(0.02)

    env.close()
    print("Done!")

if __name__ == "__main__":
    main()
