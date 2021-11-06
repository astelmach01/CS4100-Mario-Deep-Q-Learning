"""Methods for playing the game from the value iteration agent."""
import json

import gym
from nes_py.wrappers import JoypadSpace

import Q_Agent.QLearningAgent
from Q_Agent.QLearningAgent import ValueIterationAgent
import copy
episodes = 100


def play_q(env: JoypadSpace, args, actions):
    for _ in range(episodes):

        agent = ValueIterationAgent(env)
        environment = JoypadSpace(gym.make(args.env), actions)
        environment.reset()

        try:
            done = False
            while not done:
                if done:
                    _ = environment.reset()

                # double check this actually gets right action from dict
                action = agent.getAction(environment)
                _, _, done, info = environment.step(action)
                environment.render()
        except KeyboardInterrupt:
            pass
        # close the environment
        env.close()
