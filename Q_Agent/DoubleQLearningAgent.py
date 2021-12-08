from Q_Agent.QLearningAgent import ValueIterationAgent

class DoubleQLearningAgent:

    def __init__(self, env, actions, alpha=.1, gamma=.9, exploration_rate=1, exploration_rate_min=.1,
                 exploration_rate_decay=0.999999972, iterations=10000):
        self.env = env
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.exploration_rate = exploration_rate
        self.exploration_rate_min = exploration_rate_min
        self.exploration_rate_decay = exploration_rate_decay
        self.iterations = iterations

        self.agent1 = ValueIterationAgent(env, actions)
        self.agent2 = ValueIterationAgent(env, actions)