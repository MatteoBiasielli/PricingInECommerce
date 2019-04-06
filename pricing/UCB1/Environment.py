import math

import numpy as np


class Environment:

    def __init__(self, probabilities):
        self.numOfArms = len(probabilities)
        self.probabilities = probabilities
        self.variances = [round(p * (1 - p), 2) for p in probabilities]

    # returns a reward for the pulled_arm
    def get_reward(self, pulled_arm):
        tmp = np.random.binomial(1, self.probabilities[pulled_arm])
        return [tmp, self.probabilities[pulled_arm]]

    def get_best_reward(self):
        return max(self.probabilities)

    # returns a reward for each arm
    def get_realizations(self):
        tmp = np.array(np.random.binomial(1, self.probabilities))
        return tmp

    def get_probabilities(self):
        return self.probabilities.copy()

    def get_variances(self):
        return self.variances.copy()
