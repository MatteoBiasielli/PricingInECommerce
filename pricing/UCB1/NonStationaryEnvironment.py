import numpy as np
from pricing.UCB1.Environment import *


class NonStationaryEnvironment(Environment):

    def __init__(self, probabilities, horizon):
        super().__init__(probabilities)
        self.horizon = horizon
        self.t = -1
        self.clairvoyant_arm = []

    def get_reward(self, pulled_arm):

        self.t += 1
        phase_len = self.horizon / len(self.probabilities)
        phase_nr = int(self.t / phase_len)
        self.clairvoyant_arm.append(np.max(self.probabilities[phase_nr]))
        return [np.random.binomial(1, self.probabilities[phase_nr][pulled_arm]), self.probabilities[phase_nr][pulled_arm]]

    def get_realizations(self):
        tmp = np.array(np.random.binomial(1, self.probabilities[0]))
        return tmp
