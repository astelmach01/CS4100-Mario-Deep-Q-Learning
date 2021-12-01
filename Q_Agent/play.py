"""Methods for playing the game from the value iteration agent."""

from gym_super_mario_bros.actions import RIGHT_ONLY
import gym_super_mario_bros
import numpy as np
import time
import gym
from nes_py.wrappers import JoypadSpace

from Q_Agent.QLearningAgent import *
from Q_Agent.DeepQLearningAgent import *

episodes = 100


def play_q(env: JoypadSpace, args, actions):
    """Play the game using the Q-learning agent."""
    agent: ValueIterationAgent = ValueIterationAgent(env, actions)

    for _ in range(episodes):
        
        environment = None
        if actions is None:
            actions = env.action_space.n
        else:
            environment = JoypadSpace(gym.make(args.env), actions)
            environment.reset()

            
        


        done = False
        _, _ , _, info, = environment.step(0)
        state = make_state(info)
        while not done:
            if done:
                _ = environment.reset()

            action = agent.get_action(state)
            next_state, _, done, info = environment.step(action)
            state = make_state(info)
            environment.render()

        # close the environment
        env.close()


def play_deep_q(env):
    

    # Build env (first level, right only)
    # Parameters
    states = (84, 84, 4)
    actions = env.action_space.n

    # Agent
    agent = DQNAgent(states=states, actions=actions,
                    max_memory=100000, double_q=True)

    # Episodes
    episodes = 10000
    rewards = []

    # Timing
    start = time.time()
    step = 0

    # Main loop
    for e in range(episodes):

        # Reset env
        state = env.reset()

        # Reward
        total_reward = 0
        iter = 0

        # Play
        while True:

            # Show env (diabled)
            # env.render()

            # Run agent
            action = agent.run(state=state)

            # Perform action
            next_state, reward, done, info = env.step(action=action)

            # Remember transition
            agent.add(experience=(state, next_state, action, reward, done))

            # Update agent
            agent.learn()

            # Total reward
            total_reward += reward

            # Update state
            state = next_state

            # Increment
            iter += 1

            # If done break loop
            if done or info['flag_get']:
                break

        # Rewards
        rewards.append(total_reward / iter)

        # Print
        if e % 100 == 0:
            print('Episode {e} - +'
                'Frame {f} - +'
                'Frames/sec {fs} - +'
                'Epsilon {eps} - +'
                'Mean Reward {r}'.format(e=e,
                                        f=agent.step,
                                        fs=np.round(
                                            (agent.step - step) / (time.time() - start)),
                                        eps=np.round(agent.eps, 4),
                                        r=np.mean(rewards[-100:])))
            start = time.time()
            step = agent.step

    # Save rewards
    np.save('rewards.npy', rewards)
