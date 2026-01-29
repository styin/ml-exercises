import os
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

class PointMassEnv(MujocoEnv, utils.EzPickle):
    """
    A simple Point Mass environment.
    
    The agent controls a point mass (green sphere) in 2D space.
    The goal is to move to the red target.
    
    INHERITANCE:
    - MujocoEnv: The base class for all MuJoCo environments. It handles:
        - Loading the XML model
        - Creating the physics simulation (self.model, self.data)
        - Providing the render() method
        - Providing the simulation loop abstraction
    - utils.EzPickle: A mixin that allows the environment to be serialized (pickled).
      This is required if you want to save the environment state or use it with 
      multiprocessing (e.g., SubprocVecEnv in Stable-Baselines3).
    
    Observation Space (6 dimensions):
        - 0,1: Agent Position (x, y)
        - 2,3: Agent Velocity (vx, vy)
        - 4,5: Target Position (tx, ty)
        
    Action Space (2 dimensions):
        - 0: Force in X direction (clipped to [-1, 1])
        - 1: Force in Y direction (clipped to [-1, 1])
        
    Reward Function:
        - Dense reward: -distance_to_target (guides agent to target)
        - Control cost: -sum(action^2) (penalizes high energy usage)
    """
    
    metadata = {
        "render_modes": [
            "human",      # Render to a native window
            "rgb_array",  # Return an image array (for video recording)
            "depth_array",# Return a depth map
        ],
        "render_fps": 50, # Used by the renderer to determine playback speed
    }

    def __init__(self, render_mode=None, **kwargs):
        # 1. Initialize EzPickle first (important for serialization)
        utils.EzPickle.__init__(self, render_mode, **kwargs)
        
        # 2. Locate the MuJoCo XML file
        # It's best practice to keep assets relative to the python file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, "assets", "point_mass.xml")
        
        # 3. Define Observation Space
        # We manually define this to match what _get_obs() returns.
        # Shape is (6,) because: pos(2) + vel(2) + target(2)
        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )
        
        # 4. Initialize MujocoEnv
        # frame_skip=2: This is a crucial parameter for simulation speed.
        # The XML defines timestep="0.01" (10ms).
        # frame_skip=2 means that for every call to env.step(), the physics engine
        # will advance 2 * 0.01 = 0.02 seconds.
        # This gives us a control frequency of 1/0.02 = 50Hz.
        MujocoEnv.__init__(
            self,
            model_path=xml_path,
            frame_skip=2,
            observation_space=observation_space,
            render_mode=render_mode,
            default_camera_config={"trackbodyid": 0}, # Camera follows body ID 0 (the torso)
            **kwargs,
        )

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.
        
        Args:
            action (np.ndarray): An array of shape (2,) containing forces [fx, fy].
            
        Returns:
            observation (np.ndarray): The agent's observation of the current state.
            reward (float): The amount of reward returned step.
            terminated (bool): Whether the episode has ended naturally (goal reached).
            truncated (bool): Whether the episode ended early (time limit/out of bounds).
            info (dict): Auxiliary diagnostic information.
        """
        # 1. Valid range enforcement
        # Ensure action stays within valid bounds [-1, 1] defined in XML actuators
        action = np.clip(action, -1.0, 1.0)
        
        # 2. Record state before simulation (optional)
        # self.data.qpos contains the joint positions (generalized coordinates)
        # We assume indices 0,1 correspond to the agent's x,y position
        xpos_before = self.data.qpos[:2].copy()
        
        # 3. Physics Simulation
        # This function calls mujoco.mj_step() 'self.frame_skip' times.
        # It takes the 'action' and applies it to the actuators.
        self.do_simulation(action, self.frame_skip)
        
        # 4. Get state after simulation
        xpos_after = self.data.qpos[:2].copy()
        
        # 5. Get target position
        # We access the target body by name "target"
        target_pos = self.data.body("target").xpos[:2]
        
        # 6. Calculate Reward
        # Pure distance-based reward (negative distance)
        dist_to_target = np.linalg.norm(xpos_after - target_pos)
        reward_dist = -dist_to_target
        
        # Control penalty (minimize energy usage / erratic movements)
        # Small coefficient (0.1) keeps it secondary to the main objective
        reward_ctrl = -np.square(action).sum() * 0.1
        
        # Total reward is the sum of components
        reward = reward_dist + reward_ctrl
        
        # 7. Check formatting (Termination)
        # Terminated: The task is successfully completed
        terminated = bool(dist_to_target < 0.05)
        
        # Truncated: The episode was cut short (e.g. by time limit or violation)
        # Here we check if the agent went out of the [-10, 10] arena bounds
        truncated = False
        if abs(xpos_after[0]) > 9.5 or abs(xpos_after[1]) > 9.5:
            truncated = True
        
        # 8. Construct Observation
        observation = self._get_obs()
        
        # 9. Render frame (if mode is "human")
        if self.render_mode == "human":
            self.render()
            
        # 10. Return standard Gym tuple
        return observation, reward, terminated, truncated, dict(
            reward_dist=reward_dist,
            reward_ctrl=reward_ctrl,
            distance=dist_to_target
        )

    def _get_obs(self):
        """
        Constructs the observation vector.
        
        We must extract relevant physics state from `self.data` and
        format it into a single flat numpy array.
        """
        # qpos: Joint positions (x, y)
        position = self.data.qpos.flat.copy()
        
        # qvel: Joint velocities (vx, vy)
        velocity = self.data.qvel.flat.copy()
        
        # Target position from body coordinates (not joints)
        target_pos = self.data.body("target").xpos[:2]
        
        # Concatenate: [agent_pos_x, agent_pos_y, agent_vel_x, agent_vel_y, target_x, target_y]
        return np.concatenate([position, velocity, target_pos])

    def reset_model(self):
        """
        Resets the environment to an initial state.
        
        This is where we apply Domain Randomization to make the policy robust.
        We randomize the starting position of the agent and the target.
        """
        # 1. Randomize agent position (qpos)
        # self.init_qpos is the initial position from the XML file
        # model.nq is number of generalized coordinates (2 in this case)
        qpos = self.init_qpos + self.np_random.uniform(
            low=-2.0, high=2.0, size=self.model.nq
        )
        
        # 2. Randomize agent velocity (qvel)
        # Start with a small random velocity to aid exploration
        # model.nv is number of degrees of freedom (2 in this case)
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.1, high=0.1, size=self.model.nv
        )
        
        # 3. Randomize target position
        # We want the target to appear anywhere in the 10x10 arena
        target_x = self.np_random.uniform(low=-5.0, high=5.0)
        target_y = self.np_random.uniform(low=-5.0, high=5.0)
        
        # MANIPULATING BODIES DIRECTLY:
        # Since 'target' is a body without joints, it's not in qpos.
        # To move it during reset, we modify the `body_pos` field in the
        # model structure directly. This changes its default position.
        target_id = self.model.body("target").id
        self.model.body_pos[target_id][:2] = [target_x, target_y]
        
        # 4. Set the physics state
        # This function resets the simulation to these qpos and qvel values
        self.set_state(qpos, qvel)
        
        # 5. Return the first observation
        return self._get_obs()
