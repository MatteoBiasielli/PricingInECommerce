import numpy as np


class Environment:

    def __init__(self, probabilities):
        self.numOfArms = len(probabilities)
        self.probabilities = probabilities

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

    def get_highest_variance(self, marginal_profits=None):
        assert all(isinstance(x, (int, float)) for x in self.probabilities)
        variances = []
        if marginal_profits is None:
            variances = [round(p * (1 - p), 2) for p in self.probabilities]
        else:
            for prob, mp in zip(self.probabilities, marginal_profits):
                u = prob * mp
                var = round((((0 - u) ** 2) * (1 - prob) + ((mp - u) ** 2) * prob), 2)
                variances.append(var)

        return max(variances)
