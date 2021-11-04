"""Methods for playing the game from the value iteration agent."""
from Q_Agent.QLearningAgent import ValueIterationAgent

episodes = 100


def play_q(env):
    """
    Play the environment making uniformly random decisions.

    Args:
        env (gym.Env): the initialized gym environment to play
        steps (int): the number of random steps to take

    Returns:
        None

    """
    for _ in range(episodes):
        state = env.reset()
        epochs, penalties, reward = 0, 0, 0
        agent = ValueIterationAgent(env)
        try:
            state = env.reset()
            done = False
            while not done:
                if done:
                    _ = env.reset()
                action = agent.getAction(state)
                state, reward, done, info = env.step(action)
                env.render()
        except KeyboardInterrupt:
            pass
        # close the environment
        env.close()
