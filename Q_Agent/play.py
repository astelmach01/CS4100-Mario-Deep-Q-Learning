"""Methods for playing the game from the value iteration agent."""
import Q_Agent.QLearningAgent
from Q_Agent.QLearningAgent import ValueIterationAgent

episodes = 100


def play_q(env):
    for _ in range(episodes):

        agent = ValueIterationAgent(env)
        try:
            done = False
            while not done:
                if done:
                    _ = env.reset()
                action = agent.getAction()
                state, reward, done, info = env.step(action)
                env.render()
        except KeyboardInterrupt:
            pass
        # close the environment
        env.close()
