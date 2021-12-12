from Q_Agent.QLearningAgent import ValueIterationAgent
import random
from Q_Agent.DeepQLearningAgent import SkipFrame
import numpy as np
import json

file_name = "C:\\Users\Andrew Stelmach\Desktop\Mario Q Learning\\q_tables"

def make_state(info):
    return str(info["x_pos"]) + "," + str(info["y_pos"])

class DoubleQLearningAgent(ValueIterationAgent):

    def __init__(self, env, actions, alpha=.1, gamma=.9, exploration_rate=1, exploration_rate_min=.1,
                 exploration_rate_decay=0.99999, iterations=10000):
        self.env = SkipFrame(env, skip=5)
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay
        self.iterations = iterations

        self.agent1 = ValueIterationAgent(env, actions)
        self.agent2 = ValueIterationAgent(env, actions)
        self.valueIteration()

    def epsilon_greedy_action(self, state):
        if random.uniform(0, 1) < self.exploration_rate:
            action = random.randint(0, self.env.action_space.n - 1)
        else:
            action = self.get_action(state)

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        return action

    def get_action(self, state):
        # get action from sum of Q1 and Q2
        summed_q_values = self.agent1.q_values[state] + self.agent2.q_values[state]
        return np.argmax(summed_q_values)


    def valueIteration(self):
        print(self.env.get_keys_to_action())
        print("number of actions: " + str(self.env.action_space.n))
        # print(self.q_values)
        #   help(self.env.unwrapped)

        # For plotting metrics
        num_done_well = 0

        # keeping track of the x values we've hit
        x_s = set()
        # changed reward range to -100, 100
        for i in range(1, self.iterations):
            state = self.env.reset()
            next_state, reward, done, info = self.env.step(0)

            done = False  # if you died and have 0 lives left

            # used to end game early
            iteration = 1

            while not done:

                # choose action
                action = self.epsilon_greedy_action(state)

                next_state, reward, done, info = self.env.step(action)

                next_state = make_state(info)

                # check if you've been in same x position for a while
                # and if so, end game early
                # if iteration % 50 == 0:
                #     if detect == info["x_pos"]:
                #         # reward *= -2
                #         done = True
                #     detect = info["x_pos"]

                # update one agent randomly
                if random.uniform(0, 1) < 0.5:
                    next_max = self.agent2.get_max_value(next_state)
                    self.agent1.updateQValue(reward, state, action, next_max)
                else:
                    next_max = self.agent1.get_max_value(next_state)
                    self.agent2.updateQValue(reward, state, action, next_max)

                state = next_state
                iteration += 1

                if i > self.iterations / 2:
                    self.env.render()

                # amount of times we've gotten past 3rd pipe
                if info["x_pos"] > 1400:
                    num_done_well += 1
                    
                if info["x_pos"] > 3000:
                    print()
                    print("BEAT LEVEL")
                    print()

                x_s.add(info["x_pos"])
                

            print("Iteration " + str(i) + ": x_pos = " + str(info["x_pos"]) + ". Reward: " + str(
                reward) + ". Q-value 1: " + str(self.agent1.q_values[state][action]) +". Q-value 2: "
                  + str(self.agent2.q_values[state][action]) +
                  ". Epsilon: " + str(
                self.exploration_rate))

        print("Training finished.\n")
        print("Largest x_pos: " + str(max(x_s)))
        print("Num done well: " + str(num_done_well))

        # write q table to file
        self.agent1.q_values = dict((''.join(str(k)), str(v)) for k, v in self.q_values.items())
        self.agent2.q_values = dict((''.join(str(k)), str(v)) for k, v in self.q_values.items())


        try:
            with open(file_name + "1st_q_table", 'w') as convert_file:
                convert_file.write(json.dumps(self.agent1.q_values))
        except:
            q = 2

        try:
            with open(file_name + "2nd_q_table", 'w') as convert_file:
                convert_file.write(json.dumps(self.agent2.q_values))
        except:
            q = 2

        with open(file_name + "x_s.txt", 'w') as f:
            for item in x_s:
                f.write("%s\n" % item)

