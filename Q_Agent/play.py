"""Methods for playing the game from the value iteration agent."""

import gym
from nes_py.wrappers import JoypadSpace

from Q_Agent.QLearningAgent import ValueIterationAgent
import copy
episodes = 100


def play_q(env: JoypadSpace, args, actions):
    
    agent = ValueIterationAgent(env)
    
    for _ in range(episodes):
        
        environment = None
        if actions == None:
            actions = env.action_space.n
        else:
            environment = JoypadSpace(gym.make(args.env), actions)
            environment.reset()
            
        

        try:
            done = False
            while not done:
                if done:
                    _ = environment.reset()

                action = agent.getAction(environment)
                _, _, done, info = environment.step(action)
                environment.render()
        except KeyboardInterrupt:
            pass
        # close the environment
        env.close()
