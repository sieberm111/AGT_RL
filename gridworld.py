import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC, PPO, A2C, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


class CustGrid(gym.Env):
    def __init__(self, windy=False):
        super(CustGrid, self).__init__()

        self.size = [3, 4]
        self.windy = windy

        self.observation_space = spaces.Discrete(11)
        self.action_space = spaces.Discrete(4)

        self.max_steps = 1000


        self.agent_location = [2, 0]
        self.fire_pit = [1, 3]
        self.target_location = [0, 3]
        self.obstruction_location = [1, 1]

        self.action_to_direction = {
            0: np.array([1, 0]),  # down
            1: np.array([0, 1]),  # righ
            2: np.array([-1, 0]),  # up
            3: np.array([0, -1]),  # left
        }

        self.agent_to_obs = {(2, 0): 0,
                              (1, 0): 1,
                              (0, 0): 2,
                              (0, 1): 3,
                              (0, 2): 4,
                              (0, 3): 5,
                              (2, 1): 6,
                              (2, 2): 7,
                              (2, 3): 8,
                              (1, 2): 9,
                              (1, 3): 10}
    def step(self, action):
        if self.windy:
            if np.random.rand()>0.5:
                action = np.random.randint(0, 4)
                direction = self.action_to_direction[action]
            else:
                direction = self.action_to_direction[action]

        # Firstly check if we are not blocked by obstruction
        if np.array_equal(self.agent_location + direction, self.obstruction_location):
            pass
        else:  # We use `np.clip` to make sure we don't leave the grid
            self.agent_location = [ \
                np.clip(self.agent_location[0] + direction[0], 0, self.size[0] - 1), \
                np.clip(self.agent_location[1] + direction[1], 0, self.size[1] - 1)]

        reward, done = self.get_reward()

        observation = self.agent_to_obs[tuple(self.agent_location)]
        info = {}

        if self.max_steps == 0:
            terminated = True
        else:
            self.max_steps -= 1
            terminated = False

        return observation, float(reward), done, terminated, info
    def reset(self, seed=0):
        self.agent_location = [2, 0]
        self.fire_pit = [1, 3]
        self.target_location = [0, 3]
        self.obstruction_location = [1, 1]
        self.max_steps = 100

        observation = self.agent_to_obs[tuple(self.agent_location)]
        info = {}

        return observation, info
    def render(self):
        world = np.zeros(self.size)
        world[tuple(self.fire_pit)] = 5
        world[tuple(self.obstruction_location)] = 2
        world[tuple(self.target_location)] = 3
        world[tuple(self.agent_location)] = 1

        anotation = [['land'] * 5 for i in range(3)]
        anotation[self.fire_pit[0]][self.fire_pit[1]] = 'fire'
        anotation[self.obstruction_location[0]][self.obstruction_location[1]] = 'obstruction'
        anotation[self.target_location[0]][self.target_location[1]] = 'target'
        anotation[self.agent_location[0]][self.agent_location[1]] = 'agent'

        fig, ax = plt.subplots()
        ax.matshow(world, cmap='seismic')

        for (i, j), z in np.ndenumerate(world):
            ax.text(j, i, anotation[i][j], ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
        plt.show()

    def get_reward(self):
        # An episode is done if the agent has reached the target or is in fire pit
        terminated = np.array_equal(self.agent_location, self.target_location)
        dead = np.array_equal(self.agent_location, self.fire_pit)

        if terminated:  # Binary sparse rewards
            reward = 1  # + self.step_cost
            end = True
        elif dead:
            reward = -1 # + self.step_cost
            end = True
        else:
            # reward = -np.linalg.norm(
            #     np.subtract(self.agent_location, self.target_location), ord=1 )
            reward = 0
            end = False
        return reward, end


if __name__ == '__main__':

    env = CustGrid(windy=True)
    check_env(env)

    done = False
    # while not done:
    #     action = env.action_space.sample()
    #     obs, reward, done, terminated, info = env.step(action)
    #     print(obs, reward, done, terminated, info)


    model = PPO('MlpPolicy', env, verbose=1)
    print(evaluate_policy(model, env, n_eval_episodes=10))
    model.learn(10000, log_interval=1)
    model.save("ppo_gridworld_sparse")
    print(evaluate_policy(model, env, n_eval_episodes=10))
    #
    #
    model = PPO('MlpPolicy', env, verbose=1)
    obs, info = env.reset()
    model = model.load("ppo_gridworld_sparse.zip")

    while True:
        action = model.predict(obs, deterministic=False)
        obs, reward, done, terminated, info = env.step(int(action[0]))
        env.render()
        if done == True:
            break
