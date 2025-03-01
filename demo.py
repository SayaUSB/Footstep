import time
from gymnasium.envs.registration import register
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from rl_zoo3.utils import get_model_path
import gym_footsteps_planning
import gymnasium

register(
    id="footsteps-planning-any-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningAnyEnv",
    max_episode_steps=1000,  # Set maximum episode length to 50 steps for demonstration purposes. In a real-world scenario, this value should be set to the maximum possible number of steps in an episode.
)
env = gymnasium.make("footsteps-planning-any-v0")
env = gymnasium.wrappers.TimeLimit(env, max_episode_steps=1000)
env.reset()
step = 0

while True:
    state, reward, done, truncated, infos = env.step([0.1, 0.0, 0.2])
    step += 1

    print(f"STEP [{step}]")
    # print(f"State: {state}")
    # print(f"Reward: {reward}, Done: {done}, Truncated: {truncated}")

    env.render()

    if done or truncated:
        step = 0
        env.reset()

    time.sleep(0.05)
