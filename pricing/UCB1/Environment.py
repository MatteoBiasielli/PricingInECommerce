import numpy as np


class Environment:

    def __init__(self, probabilities):
        self.numOfArms = len(probabilities)
        self.probabilities = probabilities

    # Given a pulled_arm, returns a realization and the expected reward relative to that precise arm
    def get_reward(self, pulled_arm):
        tmp = np.random.binomial(1, self.probabilities[pulled_arm])
        return [tmp, self.probabilities[pulled_arm]]

    # Returns a realization for each arm
    def get_realizations(self):
        tmp = np.array(np.random.binomial(1, self.probabilities))
        return tmp
