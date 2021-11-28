import copy
import json
import math
from os import environ
import random
import numpy as np

from nes_py.wrappers import JoypadSpace

from Q_Agent.util import Counter

'''
Things you tried
-removed time space
- changed state
-custom reward function
-changed bounds of reward
-special rewards
-changed action space
'''

# TODO: actions are actually right and jump not simple
file_name = 'q_tables\custom_score_right_high_alpha.txt'


def make_state(info):
    return str(info["x_pos"]) + " , " + str(info["y_pos"])


def custom_reward(info: dict):
    if info["flag_get"]:
        return 200000

    total = 0
    total += 1 / 100 * (1 / 1000 * ((info["x_pos"] - 40) ** 2))
    total += info['score'] / 100
    total += info['coins'] * 10

    return total


class ValueIterationAgent:

    def __init__(self, env: JoypadSpace, actions, alpha=.5, gamma=.95, exploration_rate=1, exploration_rate_min=.1,
                 exploration_rate_decay=0.99999975, iterations=7500):

        self.env: JoypadSpace = env
        self.actions = actions

        self.alpha = alpha
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay
        self.iterations = iterations
        self.max_steps_per_hold = 50  # hold down button for [0, max_steps_per_hold] frames

        # first value = if u should take action of holding down button
        # second value = whatever button ur holding down
        # third value = how many frames u should hold down
        self.holding_down = (False, None, self.max_steps_per_hold)

        self.prev_score = 0

        # try to load in q table from previously written text file
        # try:
        #     values = json.load(open(file_name))
        #     self.q_values = Counter()
        #     for key, value in values.items():
        #         self.q_values[key] = float(value)
        # except:
        #     self.q_values = Counter()

        self.q_values = Counter(self.env.action_space.n)

        self.valueIteration()

    def custom_score_reward(self, info, reward):
        reward += (info['score'] - self.prev_score) / 40.0
        self.prev_score = info['score']

        return reward

    def epsilon_greedy_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randrange(0, self.env.action_space.n)
        else:
            action = self.get_action(state)
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        return action

    def valueIteration(self):

        print(self.env.get_keys_to_action())
        print("number of actions: " + str(self.env.action_space.n))
        # print(self.q_values)
        #   help(self.env.unwrapped)

        # For plotting metrics
        epochs = []
        num_done_well = 0

        # keeping track of the x values we've hit
        x_s = set()
        # changed reward range to -100, 100
        for i in range(1, self.iterations):
            state = self.env.reset()

            done = False  # if you died and have 0 lives left

            # used to end game early
            iteration = 1
            detect = -1

            while not done:

                # choose action
                action = self.epsilon_greedy_action(state)

                next_state, reward, done, info = self.env.step(action)

                next_state = make_state(info)
                reward = round(custom_reward(info), 5)

                # check if you've been in same x position for a while
                # and if so, end game early
                # if iteration % 50 == 0:
                #     if detect == info["x_pos"]:
                #         # reward *= -2
                #         done = True
                #     detect = info["x_pos"]

                # implement q learning
                old_value = self.q_values[state][action]
                next_max = self.get_max_value(next_state, i)

                # Q(s, a) <- Q(s, a) + alpha * (reward + discount * max(Q(s', a')) - Q(s, a))
                self.q_values[state][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

                state = next_state
                iteration += 1

                self.env.render()

                # amount of times we've gotten past 2nd pipe
                if info["x_pos"] > 600:
                    num_done_well += 1

                x_s.add(info["x_pos"])
            epochs.append((i, reward))

            print("Iteration " + str(i) + ": x_pos = " + str(info["x_pos"]) + ". Reward: " + str(
                reward) + ". Q-value: " + str(self.q_values[state][action]))

        print("Training finished.\n")
        print("Largest x_pos: " + str(max(x_s)))
        print("Num done well: " + str(num_done_well))

        # write q table to file
        self.q_values = dict((''.join(str(k)), str(v)) for k, v in self.q_values.items())
        with open(file_name, 'w') as convert_file:
            convert_file.write(json.dumps(self.q_values))

        with open(file_name + "x_s.txt", 'w') as f:
            for item in x_s:
                f.write("%s\n" % item)

    def get_action(self, state):
        x = self.q_values[state]
        return np.argmax(x)

    def get_max_value(self, state, i):
        if i > 50:
            t = 6
        x = self.q_values[state]
        return np.max(x)
