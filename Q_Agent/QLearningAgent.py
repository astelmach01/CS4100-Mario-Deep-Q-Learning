import time

import gym
import numpy as np
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from Q_Agent.util import Counter
import random
from IPython.display import clear_output
import copy
from datetime import datetime
import json


def make_state(info):
    return info["x_pos"], info["y_pos"], info["time"], info["coins"], info["status"], info["life"]


class ValueIterationAgent:
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, env, discount=0.9, iterations=1000):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)
        self.discount = discount
        self.iterations = iterations
        self.q_values = Counter()
        self.valueIteration()

    def valueIteration(self):

        actions = self.env.unwrapped.get_action_meanings()
        print(actions)
        print(self.env.get_keys_to_action())
     #   help(self.env.unwrapped)

        # Hyperparameters
        alpha = 0.1
        gamma = 0.9
        epsilon = 0.2

        # For plotting metrics
        all_epochs = []
        all_penalties = []

        for i in range(1, 101):
            state = self.env.reset()
            action = self.getAction()
            state = hash(str(state))
            start_state = hash(str(state))
            done = False

            iteration = 0
            if i == 3:
                x = 5
            while not done:

                try:
                    next_state, reward, done, info = self.env.step(action)
                    x_pos = info["x_pos"]
                except ValueError:
                    break

                if done:
                    reward = 10000

                if iteration % 10 == 0:
                    if x_pos == info["x_pos"]:
                        reward = -1000

                next_state = make_state(info)

                self.q_values[state] = reward + (gamma * self.getMaxValue())

                state = next_state
                self.env.render()

                iteration += 1
            print("starting state value: " + str(self.q_values[start_state]))
            print("reward at iteration " + str(i) + ": for 2nd to last action " + str(self.q_values[state]))

            print(datetime.now().strftime("%H:%M:%S"))

            if i % 10 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")

        print("Training finished.\n")
        with open('convert.txt', 'w') as convert_file:
            convert_file.write(json.dumps(self.q_values))

    def getAction(self):
        next_max = float('-inf')
        best_action = None
        env_copy = copy.copy(self.env)
        n = self.env.action_space.n
        for trying in range(n):
            env = env_copy
            try:
                _, _, tried, y = env.step(trying)
                if self.q_values[make_state(y), trying] > next_max:
                    next_max = self.q_values[make_state(y), trying]
                    best_action = trying
            except:
                self.env.render()
                time.sleep(1)
                continue

        return best_action

    def getMaxValue(self):
        largest = float('-inf')
        env_copy = copy.copy(self.env)
        n = self.env.action_space.n
        for trying in range(n):
            env = env_copy
            try:
                _, _, tried, y = env.step(trying)
                largest = max(largest, self.q_values[make_state(y)])
            except ValueError:
                continue

        return largest
