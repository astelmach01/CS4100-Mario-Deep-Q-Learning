"""Methods for playing the game from the value iteration agent."""
from QLearningAgent import ValueIterationAgent



def play(env):
    """
    Play the environment making uniformly random decisions.

    Args:
        env (gym.Env): the initialized gym environment to play
        steps (int): the number of random steps to take

    Returns:
        None

    """
    agent = ValueIterationAgent(env)
    try:
        done = False
        while not done:
            if done:
                _ = env.reset()
            action = agent.getPolicy(env.observation_space)
            _, reward, done, info = env.step(action)
            env.render()
    except KeyboardInterrupt:
        pass
    # close the environment
    env.close()
