import numpy as np


class Environment:

    def __init__(self, probabilities):
        self.numOfArms = len(probabilities)
        self.probabilities = probabilities

    # returns a reward for the pulled_arm
    def get_reward(self, pulled_arm):
        tmp = np.random.binomial(1, self.probabilities[pulled_arm])
        return [tmp, self.probabilities[pulled_arm]]

    # returns a reward for each arm
    def get_realizations(self):
        tmp = np.array(np.random.binomial(1, self.probabilities))
        return tmp

    def get_best_profit_reward(self, marginal_profits):
        return np.max(np.array(self.probabilities) * np.array(marginal_profits))

    def get_probabilities(self):
        return self.probabilities.copy()
