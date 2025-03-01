from stable_baselines3 import TD3
import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
import time

# Load the model
model = TD3.load("logs/td3/footsteps-planning-right-v0_1/best_model.zip")

# Input the starting point and end point
mulai = np.array(list(map(float, input("Mulai (x,y,yaw): ").split())), dtype=np.float32)
akhir = np.array(list(map(float, input("Akhir (x,y,yaw): ").split())), dtype=np.float32)
options = {
    "start_foot_pose": mulai,
    "target_foot_pose": akhir,
}

env = gym.make("footsteps-planning-right-v0")
obs, _ = env.reset()
options = options or {}
while True:
    action, states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated:
        break
env.close()