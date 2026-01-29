from gymnasium.envs.registration import register

from .point_mass import PointMassEnv

# Register the environment
register(
    id="PointMass-v0",
    entry_point="src.envs:PointMassEnv",
    max_episode_steps=500,
)
