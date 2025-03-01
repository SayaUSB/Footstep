import numpy as np
import onnxruntime as ort
import gymnasium as gym
from gymnasium.envs.registration import register
import time

# Load the ONNX model
session = ort.InferenceSession("footsteps_planning_right.onnx")

# Input into the ONNX model
mulai = np.array(list(map(float, input("Mulai (x, y, yaw): ").split())))
akhir = np.array(list(map(float, input("Akhir (x, y, yaw): ").split())))
register(
    id="footsteps-planning-right-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightEnv",
    max_episode_steps=50,
)

# Environment
env = gym.make("footsteps-planning-right-v0")
options = {
    "start_foot_pose": mulai,
    "target_foot_pose": akhir,
    "panjang": 20,
    "lebar": 20,
}
obs, _ = env.reset(options=options)

# ONNX Session
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

while True:
    obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
    action = session.run([output_name], {input_name: obs_input})[0]
    action = action[0]
    obs, rewards, terminated, truncated, info = env.step(action)
    print(*info['Foot Coord'])
    time.sleep(1)
    env.render()
    if terminated:
        break
env.close()