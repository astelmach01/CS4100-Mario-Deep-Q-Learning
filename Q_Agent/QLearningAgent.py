import copy
import json
import math
from os import environ
import random

from nes_py.wrappers import JoypadSpace

from Q_Agent.util import Counter

'''
Things you tried
-removed time space
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
    total += 1/100 * (1 / 1000 * ((info["x_pos"] - 40) ** 2))
    total += info['score'] / 100
    total += info['coins'] * 10

    return total


class ValueIterationAgent:

    def __init__(self, env: JoypadSpace, alpha=.5, gamma=.95, epsilon=.1, iterations=7500):

        self.env: JoypadSpace = env
        self.alpha: float = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.iterations = iterations
        self.max_steps_per_hold = 15
        self.holding_down = (False, None, self.max_steps_per_hold)

        self.prev_score = 0

        # try to load in q table from previously written text file
        try:
            values = json.load(open(file_name))
            self.q_values = Counter()
            for key, value in values.items():
                self.q_values[key] = float(value)
        except:
            self.q_values = Counter()

        self.valueIteration()

    def custom_score_reward(self, info, reward):
        reward += (info['score'] - self.prev_score) / 40.0
        self.prev_score = info['score']

        return reward

    def epsilon_greedy_action(self):
        # epsilon greedy
        if random.uniform(0, 1) < self.epsilon:
            return random.randrange(0, self.env.action_space.n)
        else:
            return self.getAction()

    def valueIteration(self):

        actions = self.env.unwrapped.get_action_meanings()
        print(actions)
        print(self.env.get_keys_to_action())
        print("number of actions: " + str(self.env.action_space.n))
        # print(self.q_values)
        #   help(self.env.unwrapped)

        # For plotting metrics
        epochs = []
        num_done_well = 0

        x_s = set()
        # changed reward range to -100, 100
        for i in range(1, self.iterations):
            state = self.env.reset()
            state = hash(str(state))
            done = False
            iteration = 1
            detect = -1

            if i == 100:
                x = 5

            while not done:

                # choose action
                if self.holding_down[0]:
                    action = self.holding_down[1]
                    if self.holding_down[2] == 0:
                        self.holding_down = (False, None, 50)
                    else:
                        self.holding_down = (True, self.holding_down[1], self.holding_down[2] - 1)
                else:
                    action = self.epsilon_greedy_action()

                try:
                    next_state, reward, done, info = self.env.step(action)
                    reward = round(custom_reward(info), 5)
                    # check values of reward for no clipping

                except:
                    break

                next_state = make_state(info)

                if iteration % 20 == 0:
                    if detect == info["x_pos"]:
                        # reward *= -2
                        done = True
                    detect = info["x_pos"]

                if info["x_pos"] > 600:
                    num_done_well += 1

                # implement q learning
                key = str((state, action))

                old_value = round(self.q_values[key], 3)

                next_max = self.getMaxValue()

                # Q(s, a) <- Q(s, a) + alpha * (reward + discount * max(Q(s', a')) - Q(s, a))
                # self.q_values[key] = old_value + alpha * (reward + gamma * next_max - old_value)
                self.q_values[key] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

                # print(self.q_values[str((state, action))])
                state = next_state
                iteration += 1

                # self.env.render()

                x_s.add(info["x_pos"])
            epochs.append((i, reward))

            if info["x_pos"] > 722:
                print("Iteration " + str(i) + ": x_pos = " +
                      str(info["x_pos"]) + ". Reward: " + str(reward))

            else:
                print("Iteration " + str(i) + ". Reward: " + str(reward))

        print("Training finished.\n")
        print("Largest x_pos: " + str(max(x_s)))
        print("Num done well: " + str(num_done_well))

        self.q_values = dict((''.join(str(k)), str(v)) for k, v in self.q_values.items())
        with open(file_name, 'w') as convert_file:
            convert_file.write(json.dumps(self.q_values))

        with open(file_name + "x_s.txt", 'w') as f:
            for item in x_s:
                f.write("%s\n" % item)

    def getAction(self, fake_env: JoypadSpace = 1):
        if fake_env == 1:
            fake_env = self.env

        next_max = float('-inf')
        best_action = 2
        env_copy = copy.copy(fake_env)

        n = self.env.action_space.n
        for trying in range(n):
            env = env_copy
            try:
                _, _, _, y = env.step(trying)
                key = str((make_state(y), trying))
                if self.q_values[key] > next_max:
                    next_max = self.q_values[key]
                    best_action = trying
            except:
                continue

        if self.try_hold(env_copy) > next_max:
            self.holding_down = (True, self.env.action_space.n - 1, 20)
            return self.env.action_space.n - 1

        return best_action

    def getMaxValue(self):
        largest = float('-inf')
        env_copy = copy.copy(self.env)

        n = self.env.action_space.n
        for trying in range(n):
            env = env_copy
            try:
                _, _, _, y = env.step(trying)
                largest = max(largest, self.q_values[str((make_state(y), trying))])
            except ValueError:
                continue

        return largest if largest > self.try_hold() else self.try_hold()

    def try_hold(self, environmnent=None):
        if environmnent is None:
            environmnent = self.env
        # TODO make this better by choosing the actions in the best holding frame amount
        env_copy = copy.copy(environmnent)
        value = 0
        for _ in range(20):
            try:
                _, _, done, y = env_copy.step(self.env.action_space.n - 1)
                value = custom_reward(y)
                if done:
                    return value
            except ValueError:
                return value
        return value
