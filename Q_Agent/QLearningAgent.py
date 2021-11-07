import copy
import json
import math
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


def make_state(info):
    return str(info["x_pos"]) + " , " + str(info["y_pos"])


def custom_reward(info: dict):
    if info["flag_get"]:
        return 200000

    total = 0
    total += (info["x_pos"] - 40) / 20
    total += info['score'] / 100
    total += info['coins'] * 2

    return total


class ValueIterationAgent:

    def __init__(self, env, discount=0.9, iterations=1000):

        self.env = env
        self.discount = discount
        self.iterations = iterations
        try:
            values = json.load(open("convert_right_and_jump2.txt"))
            self.q_values = Counter()
            for value in values.keys():
                self.q_values[value] = float(values[value])
        except:
            self.q_values = Counter()

        self.valueIteration()

    def valueIteration(self):

        actions = self.env.unwrapped.get_action_meanings()
        print(actions)
        print(self.env.get_keys_to_action())
        print("number of actions: " + str(self.env.action_space.n))
        # print(self.q_values)
        #   help(self.env.unwrapped)

        # Hyperparameters
        alpha = 1
        gamma = 0.95
        epsilon = 0.1

        # For plotting metrics
        all_epochs = []
        all_penalties = []
        num_done_well = 0

        x_s = set()
        # changed reward range to -100, 100
        for i in range(1, 500000):
            state = self.env.reset()
            state = hash(str(state))
            done = False
            iteration = 1
            detect = -1

            if i == 100:
                x = 5
            while not done:

                if random.uniform(0, 1) < epsilon:
                    action = random.randrange(0, self.env.action_space.n)

                else:
                    action = self.getAction()

                try:
                    next_state, reward, done, info = self.env.step(action)
                    reward = custom_reward(info)
                    # check values of reward for no clipping

                except:
                    break

                next_state = make_state(info)

                if iteration % 10 == 0:
                    if detect == info["x_pos"]:
                        # reward *= -2
                        done = True
                    detect = info["x_pos"]

                if info["x_pos"] > 600:
                    num_done_well += 1

                # implement q learning
                old_value = self.q_values[str((state, action))]

                next_max = self.getMaxValue()
                # Q(s, a) <- Q(s, a) + alpha * (reward + discount * max(Q(s', a')) - Q(s, a))
                self.q_values[str((state, action))] = old_value + alpha * (reward + gamma * next_max - old_value)

                # print(self.q_values[str((state, action))])
                state = next_state
                iteration += 1

                self.env.render()

                x_s.add(info["x_pos"])

            if info["x_pos"] > 600:
                print("Iteration " + str(i) + ": " +
                      str(info["x_pos"]) + ". Reward: " + str(reward))

            else:
                print("Iteration " + str(i) + ". Reward: " + str(reward))

        print("Training finished.\n")
        print("Largest x_pos: " + str(max(x_s)))
        print("Num done well: " + str(num_done_well))

        self.q_values = dict((''.join(str(k)), str(v)) for k, v in self.q_values.items())
        with open('convert_right_and_jump_custom_reward.txt', 'w') as convert_file:
            convert_file.write(json.dumps(self.q_values))

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
                if self.q_values[key] >= next_max:
                    next_max = self.q_values[key]
                    best_action = trying
            except:
                continue

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

        return largest
