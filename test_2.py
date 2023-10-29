
from pogema_ours import pogema_v0, Hard8x8, OneCriminal
from pogema_ours.animation import AnimationMonitor, AnimationConfig
from pogema_ours.a_star_policy import BatchAStarAgent
from pogema_baselines.pymarl.modules.agents.dqn_agent import DQNAgent
import torch
import numpy as np


class ARG:
    hidden_dim1 = 128
    hidden_dim2 = 64
    n_actions = 5


grid_config = OneCriminal()
grid_config.seed = 2
env = pogema_v0(grid_config=grid_config)  # seed to control the repeatablility
env = AnimationMonitor(env)

obs, info = env.reset()

agents = BatchAStarAgent()
agent = DQNAgent(len(np.array(obs[0][0]).flatten()), ARG())
agent.load_state_dict(torch.load(
    '/home/maddpg/pogema_env/pogema_baselines/pymarl/results/models/iql__2023-10-13_14-48-27/50051/agent.th'))


while True:
    # Using random policy to make actions
    # actions = env.sample_actions()
    arr_obs, dict_obs = obs
    actions = agents.act(dict_obs[1:])
    action = agent(arr_obs[0]).flatten()
    print(actions, action)
    exit()
    # print(actions)
    obs, reward, terminated, truncated, info = env.step(actions)
    # env.render()
    # print(terminated)
    if all(terminated) or all(truncated):
        break
env.save_animation("out2.svg")
env.save_animation("out-ego2.svg", AnimationConfig(egocentric_idx=0))
