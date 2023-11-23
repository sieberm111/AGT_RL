import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


class RL_trade(gym.Env):
    def __init__(self):
        super(RL_trade, self).__init__()
        pass

    def step(self, action):
        pass


    def reset(self, seed=0):
        pass


    def render(self):
        pass

    def close(self):
        pass

if __name__ == '__main__':

    env = RL_trade()
    check_env(env)