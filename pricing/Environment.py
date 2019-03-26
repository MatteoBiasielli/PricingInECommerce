import numpy as np
from pricing.UCB1.Environment import *


class NonStationaryEnvironment(Environment):

    def __init__(self, probabilities, horizon, marginal_profits=0):

        super().__init__(probabilities)
        self.horizon = horizon
        self.marginal_profits = marginal_profits
        self.t = -1
        self.clairvoyant_arm = []

    # Given a pulled_arm, returns a realization and the expected reward relative to that precise arm
    def get_reward(self, pulled_arm):

        # Increments the time step by 1, then computes the current phase
        self.t += 1
        phase_len = self.horizon / len(self.probabilities)
        phase_nr = int(self.t / phase_len)

        # Clairvoyant arm's expected reward computation at every round:
        # If marginal profits not present, appends just the mean[i] of the optimal arm "i" relative to the current round
        # If marginal profits present, appends the (mean[i] * marginal_profits[i]) of the optimal arm "i"
        # relative to the current round instead
        if self.marginal_profits == 0:
            self.clairvoyant_arm.append(np.max(self.probabilities[phase_nr]))
        else:
            self.clairvoyant_arm.append(np.max(np.array(self.probabilities[phase_nr]) * self.marginal_profits))
        print(phase_nr)
        print(pulled_arm)
        return [np.random.binomial(1, self.probabilities[phase_nr][pulled_arm]), self.probabilities[phase_nr][pulled_arm]]

    # returns a realization for each arm
    def get_realizations(self):

        tmp = np.array(np.random.binomial(1, self.probabilities[0]))
        return tmp
