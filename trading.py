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

        self.observation_space = spaces.Box(low=-1, high=1, shape=(1, 1), dtype=np.float64)
        # Buy or Sell
        self.action_space = spaces.Discrete(2)

        self.counter = 0
        self.iterator = 0
        self.prize = 0

    def step(self, action):
        # Sell
        if action == 0:
            reward = self.prize
        #Buy
        else:
            reward = -self.prize

        if self.counter == 500:
            done = True
        else:
            done = False

        self.counter += 1
        self.iterator += 0.1

        obs = np.array(np.cos(self.iterator)).reshape(1,1)
        self.prize = obs[0]

        terminated = False
        info = {}

        return obs, float(reward), done, terminated, info


        return
    def reset(self, seed=0):
        self.reward = 0
        self.iterator = 0
        self.counter = 0
        observation = np.array(np.cos(self.iterator)).reshape(1,1)
        self.prize = observation[0]


        info = {}

        return observation, info


    def render(self):
        pass

    def close(self):
        pass

if __name__ == '__main__':

    env = RL_trade()
    check_env(env)

    model = PPO('MlpPolicy', env, verbose=1)
    print(evaluate_policy(model, env, n_eval_episodes=10))
    model.learn(10000, log_interval=1)
    model.save("ppo_trading")
    print(evaluate_policy(model, env, n_eval_episodes=10))

    total_reward = 0


    # obs = env.reset()
    # while True:
    #     model.predict(obs)