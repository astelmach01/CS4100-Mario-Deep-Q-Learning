
import numpy as np
from Q_Agent.util import Counter
import random
from IPython.display import clear_output

class ValueIterationAgent():
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
        self.env = env
        self.discount = discount
        self.iterations = iterations
        self.q_values = Counter()  # A Counter is a dict with default 0
        self.valueIteration()

    def valueIteration(self):

        # Hyperparameters
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1

        # For plotting metrics
        all_epochs = []
        all_penalties = []

        for i in range(1, 100001):
            state = self.env.reset()

            epochs, penalties, reward, = 0, 0, 0
            done = False

            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample()  # Explore action space
                else:
                    # Exploit learned values
                    action = np.argmax(self.q_values[state])

                next_state, reward, done, info = self.env.step(action)

                next_max = np.max(self.q_values[next_state])

                self.q_values[state, action] = (1 - alpha) * self.q_values[state, action] + alpha * (reward + gamma * next_max)

                if reward == -10:
                    penalties += 1

                state = next_state
                epochs += 1

            if i % 100 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")

        print("Training finished.\n")
    
    def getAction(self, state):
        return self.q_values[state]
