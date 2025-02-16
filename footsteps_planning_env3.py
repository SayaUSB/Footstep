import gymnasium
import math
import numpy as np
from typing import Optional
from gymnasium import spaces
from gym_footsteps_planning.footsteps_simulator.simulator import Simulator as FootstepsSimulator
from gym_footsteps_planning.footsteps_simulator import transform as tr


class FootstepsPlanningEnv(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self, options: Optional[dict] = None, train: bool = False, visualize: bool = False, render_mode: str = "none"
    ):
        self.options = {
            # Maximum steps
            "max_dx_forward": 0.08,  # [m]
            "max_dx_backward": 0.03,  # [m]
            "max_dy": 0.04,  # [m]
            "max_dtheta": np.deg2rad(20),  # [rad]
            # Target tolerance
            "tolerance_distance": 0.05,  # [m]
            "tolerance_angle": np.deg2rad(5),  # [rad]
            # Do we include collisions with the ball?
            "has_obstacle": False,
            "obstacle_max_radius": 0.3,  # [m]
            "obstacle_radius": None,  # [m]
            "obstacle_position": np.array([0, 0], dtype=np.float32),  # [m,m]
            # Which foot is targeted (any, left or right)
            "foot": "any",
            # Foot geometry
            "foot_length": 0.14,  # [m]
            "foot_width": 0.08,  # [m]
            "feet_spacing": 0.15,  # [m]
            # Add reward shaping term
            "shaped": True,
            # If True, the goal will be sampled in a 4x4m area, else it will be fixed at (0,0)
            "multi_goal": False,
        }
        self.options.update(options or {})

        # Render mode
        self.visualize: bool = visualize
        self.render_mode: str = render_mode

        self.simulator: FootstepsSimulator = FootstepsSimulator()
        self.simulator.feet_spacing = self.options["feet_spacing"]
        self.simulator.foot_length = self.options["foot_length"]
        self.simulator.foot_width = self.options["foot_width"]

        # Maximum speed in each dimension
        self.min_step = np.array(
            [-self.options["max_dx_backward"], -self.options["max_dy"], -self.options["max_dtheta"]], dtype=np.float32
        )
        self.max_step = np.array(
            [self.options["max_dx_forward"], self.options["max_dy"], self.options["max_dtheta"]], dtype=np.float32
        )

        # Action space is target step size (dx, dy, dtheta)
        # To keep 0 as "not moving", we use maxStep instead of maxBackwardStep,
        # but the speed is clipped when stepping
        self.action_high = np.array(
            [self.options["max_dx_forward"], self.options["max_dy"], self.options["max_dtheta"]], dtype=np.float32
        )
        self.action_low = -self.action_high

        self.action_space = spaces.Box(self.action_low, self.action_high)

        # State is position and orientation, here limited in a √(2*4²)x√(2*4²)m area arround the support foot
        # and the current step size
        # - x target support foot position in the frame of the foot
        # - y target support foot position in the frame of the foot
        # - cos(theta) target support foot orientation in the frame of the foot
        # - sin(theta) target support foot orientation in the frame of the foot
        # - is the current foot the target foot ?
        # - x obstacle position in the frame of the foot
        # - y obstacle position in the frame of the foot
        # - the obstacle radius
        max_diag_env = np.sqrt(2 * (4**2))
        self.max_obstacles = 5
        # Target + flag (5 elements)
        base_low = [-max_diag_env, -max_diag_env, -1, -1, 0]
        base_high = [max_diag_env, max_diag_env, 1, 1, 1]
        
        # Obstacle params (3 elements per obstacle)
        obstacle_low = [-max_diag_env, -max_diag_env, 0] * self.max_obstacles
        obstacle_high = [max_diag_env, max_diag_env, self.options["obstacle_max_radius"] + 0.1] * self.max_obstacles
        
        # Combine (5 + 3*5 = 20 elements)
        self.state_low_goal = np.array(base_low + obstacle_low, dtype=np.float32)
        self.state_high_goal = np.array(base_high + obstacle_high, dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=self.state_low_goal,
            high=self.state_high_goal,
            dtype=np.float32
        )

        self.reset(seed=0)

    def get_observation(self) -> np.ndarray:
        """
        Builds an observation from the current internal state with 1-5 obstacles
        """
        # Get transformation matrix from support foot to world frame
        T_support_world = tr.frame_inv(tr.frame(*self.simulator.support_pose()))

        # Calculate target position/orientation in support foot's frame
        T_support_target = T_support_world @ tr.frame(*self.target_foot_pose)
        support_target = np.array([
            T_support_target[0, 2],  # x
            T_support_target[1, 2],  # y
            T_support_target[0, 0],  # cos(theta)
            T_support_target[1, 0],  # sin(theta)
        ], dtype=np.float32)

        # Handle left foot symmetry
        if self.simulator.support_foot == "left":
            support_target[1] = -support_target[1]  # Mirror y-axis
            support_target[3] = -support_target[3]  # Mirror rotation

        # Target foot flag
        is_target_foot = 1 if (self.simulator.support_foot == self.target_support_foot) else 0

        # Initialize obstacle data (max 5 obstacles)
        obstacle_info = []
        if self.options["has_obstacle"]:
            # Process each obstacle
            for obstacle_pos, obstacle_radius in self.obstacles:
                # Transform obstacle position to support foot's frame
                support_obstacle = tr.apply(T_support_world, obstacle_pos)
                
                # Mirror y-axis for left foot
                if self.simulator.support_foot == "left":
                    support_obstacle[1] = -support_obstacle[1]
                    
                obstacle_info += [support_obstacle[0], support_obstacle[1], obstacle_radius]

            # Pad with zeros to maintain fixed size (5 obstacles * 3 values)
            while len(obstacle_info) < 5 * 3:
                obstacle_info.extend([0.0, 0.0, 0.0])
        else:
            obstacle_info = [0.0] * (5 * 3)  # No obstacles case

        # Combine all components
        state = np.array(
            [*support_target, is_target_foot] + obstacle_info,
            dtype=np.float32
        )
        assert len(state) == 20, f"Observation shape mismatch: {state.shape}"
        return state

    def ellipsoid_clip(self, step: np.ndarray) -> np.ndarray:
        """
        Applying a rescale of the order in an "ellipsoid" manner. This transforms the target step to
        a point in a space where it should lie on a sphere, ensure its norm is not high than 1 and takes
        it back to the original scale.
        """
        factor = np.array(
            [
                self.options["max_dx_forward"] if step[0] >= 0 else self.options["max_dx_backward"],
                self.options["max_dy"],
                self.options["max_dtheta"],
            ],
            dtype=np.float32,
        )
        clipped_step = step / factor

        # In this space, the step norm should be <= 1
        norm = np.linalg.norm(clipped_step)
        if norm > 1:
            clipped_step /= norm

        return clipped_step * factor

    def step(self, action):
        """
        One step of the environment. Takes one step, checks for collisions with multiple obstacles,
        and determines if the goal is reached.
        """
        # Mirror action for left foot symmetry
        if self.simulator.support_foot == "left":
            action[1] = -action[1]
            action[2] = -action[2]

        # Clip and normalize the step using ellipsoid constraints
        clipped_step = self.ellipsoid_clip(
            np.clip(action, self.action_low, self.action_high)
        )

        # Execute the step in the simulator
        self.simulator.step(*clipped_step)

        # Get current observation state
        state = self.get_observation()

        # Calculate position and orientation errors
        distance = np.linalg.norm(state[:2])
        theta_error = np.arccos(np.clip(state[2], -1.0, 1.0))  # Using cos(theta) component
        is_desired_foot = state[4] == 1

        # Collision detection with multiple obstacles
        in_obstacle = False
        if self.options["has_obstacle"]:
            # Iterate through all possible obstacle slots (max 5)
            for obstacle_idx in range(5):
                # Extract obstacle parameters from state
                base_idx = 5 + 3 * obstacle_idx
                obs_x = state[base_idx]
                obs_y = state[base_idx + 1]
                obs_radius = state[base_idx + 2]

                # Skip inactive obstacles (zero-padded)
                if obs_radius <= 0:
                    continue

                # Check all four corners of the foot
                for sx in [-1, 1]:  # Length directions
                    for sy in [-1, 1]:  # Width directions
                        # Calculate foot corner position in support foot's frame
                        corner_pos = np.array([
                            sx * self.simulator.foot_length / 2,
                            sy * self.simulator.foot_width / 2
                        ], dtype=np.float32)

                        # Calculate distance to obstacle center
                        distance_to_obs = np.linalg.norm(corner_pos - np.array([obs_x, obs_y]))

                        if distance_to_obs < obs_radius:
                            in_obstacle = True
                            break  # Exit width loop
                    if in_obstacle:
                        break  # Exit length loop
                if in_obstacle:
                    break  # Exit obstacle loop

        # Determine termination and reward
        if (distance < self.options["tolerance_distance"] and
            theta_error < self.options["tolerance_angle"] and
            is_desired_foot):
            reward = 100
            terminated = True
        else:
            reward = -0.1  # Reduced per-step penalty
            # Apply collision penalty
            if in_obstacle:
                reward -= 20.0  # Less severe than before

            # Apply shaped reward based on progress
            if self.options["shaped"]:
                # Calculate improvement from previous state
                prev_distance = np.linalg.norm(self.previous_state[:2])
                prev_theta_error = np.arccos(np.clip(self.previous_state[2], -1.0, 1.0))

                # Reward for distance reduction and angle improvement
                distance_diff = prev_distance - distance
                theta_diff = prev_theta_error - theta_error
                reward += (distance_diff * 5.0 + theta_diff * 0.5)  # Increased scaling

            terminated = False

        self.previous_state = state.copy()

        # Render if visualization is enabled
        if self.visualize:
            self.do_render()

        return state, reward, terminated, False, {}

    def reset(self, seed: int | None = None, options: Optional[dict] = None):
        """
        Resets the environment to a given foot pose and support foot
        """
        # Seeding the environment
        super().reset(seed=seed)
        options = options or {}

        # Getting the options
        start_foot_pose = options.get("start_foot_pose", None)
        start_support_foot = options.get("start_support_foot", None)
        self.target_foot_pose = options.get("target_foot_pose", None)
        self.target_support_foot = options.get("target_support_foot", None)
        self.obstacle_radius = options.get("obstacle_radius", None)

        # Choosing obstacle radius and position
        if self.options["has_obstacle"]:
            self.simulator.clear_obstacles()
            num_obstacles = self.np_random.integers(1, 6)  # Random between 1-5
            self.obstacles = []
            for _ in range(num_obstacles):
                # Random position within a 4x4m area
                obstacle_pos = self.np_random.uniform(low=-2.0, high=2.0, size=2).astype(np.float32)
                obstacle_radius = self.np_random.uniform(low=0.0, high=self.options["obstacle_max_radius"])
                self.obstacles.append((obstacle_pos, obstacle_radius))
                self.simulator.add_obstacle(obstacle_pos, obstacle_radius)

        # Choosing starting foot
        if start_support_foot is None:
            start_support_foot = "left" if (self.np_random.uniform(0, 1) > 0.5) else "right"

        # Choosing target foot
        if self.target_support_foot is None:
            if self.options["foot"] != "any":
                self.target_support_foot = self.options["foot"]
            else:
                self.target_support_foot = "left" if (self.np_random.uniform(0, 1) > 0.5) else "right"

        # Sampling starting position and orientation
        if start_foot_pose is None:
            start_foot_pose = self.np_random.uniform([-2, -2, -math.pi], [2, 2, math.pi])

        # Initializing the simulator
        self.simulator.init(*start_foot_pose, start_support_foot)

        if self.target_foot_pose is None:
            if self.options["multi_goal"]:
                self.target_foot_pose = self.np_random.uniform([-2, -2, -math.pi], [2, 2, math.pi])
            else:
                self.target_foot_pose = np.array([0, 0, 0], dtype=np.float32)

        self.simulator.set_desired_goal(*self.target_foot_pose, self.target_support_foot)
        self.previous_state = self.get_observation()
        return self.get_observation(), {}

    def render(self):
        """
        Renders the footsteps
        """
        self.visualize = True

    def do_render(self):
        self.simulator.render()
