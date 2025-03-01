from openvino.runtime import Core
import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register

# Load the OpenVino model
core = Core()
model = core.read_model("openvino_model/footsteps_planning_right.xml")
compiled_model = core.compile_model(model, "CPU")
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)


# Input into the OpenVino model
mulai = np.array(list(map(float, input("Mulai (x, y, yaw): ").split())))
akhir = np.array(list(map(float, input("Akhir (x, y, yaw): ").split())))
options = {
    "start_foot_pose": mulai,
    "target_foot_pose": akhir,
    "panjang": 8,
    "lebar": 6,
}

# Register the environment
register(
    id="footsteps-planning-right-v0",
    entry_point="gym_footsteps_planning.envs:FootstepsPlanningRightEnv",
    max_episode_steps=100,
)
env = gym.make("footsteps-planning-right-v0")
obs, _ = env.reset(options=options)

# OpenVino session
while True:
    obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
    result = compiled_model([obs_input])[output_layer]
    action = np.squeeze(result, axis=0)
    obs, rewards, terminated, truncated, info = env.step(action)
    print(*info["Foot Coord"])
    env.render()
    if terminated:
        break
env.close()
