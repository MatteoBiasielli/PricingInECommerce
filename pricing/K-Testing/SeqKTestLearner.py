import math
import queue
import random

import numpy as np
import scipy.stats as stats

from pricing.Environment import Environment


class SeqKTestLearner:

    def __init__(self, environment, num_of_candidates, marginal_profits=None, alpha=0.05,
                 plot_avg_reward=True, plot_avg_regret=True):
        self.environment = environment
        self.num_of_candidates = num_of_candidates
        if marginal_profits is not None:
            assert num_of_candidates == len(
                marginal_profits), "The marginal profits must be equal to the number of candidates!"
            self.profitMaximization = True
        else:
            self.profitMaximization = False
        self.marginal_profits = marginal_profits
        self.alpha = alpha
        self.plot_avg_reward = plot_avg_reward
        self.plot_avg_regret = plot_avg_regret
        self.n_samples_per_candidate = 100  # TODO implement formula for n
        # each row refers to the collected samples of one candidate
        self.samples = np.empty((num_of_candidates, self.n_samples_per_candidate))

    def start(self):
        # randomize candidate order
        candidates = list(range(self.num_of_candidates))
        random.shuffle(candidates)
        q = queue.Queue()
        for c in candidates:
            q.put(c)
        # get the first two candidates to test
        control = q.get()
        variation = q.get()
        # perform all tests, the last control is the winner
        for i in range(self.num_of_candidates - 1):
            # the winner becomes the control
            control = self.__test_against(variation=variation, control=control)
            # the next variation is the next unseen candidate
            if not q.empty():
                variation = q.get()

        return control

    def __test_against(self, variation, control):
        # for both collect samples
        self.__collect_samples(control)
        self.__collect_samples(variation)
        # apply hypothesis test (test if mean of variation is higher than mean of control)
        if self.__hypothesis_test(variation, control) is True:
            # variation is better than control
            return variation
        else:
            # there is no evidence that variation is better than control
            return control

    def __collect_samples(self, candidate):
        # collect the samples (buy / not buy)
        for i in range(self.n_samples_per_candidate):
            realization = self.environment.get_reward(candidate)[0]
            self.samples[candidate, i] = realization

        if self.profitMaximization:
            # turn binary samples to profit samples
            self.samples[candidate, :] = self.samples[candidate, :] * self.marginal_profits[candidate]

    def __hypothesis_test(self, c_1, c_2):
        """Performs an hypothesis test where the null hypothesis is: H_0 = u_c1 - u_c2 = 0 and the alternative one
        is: H_1 = u_c1 - u_c2 > 0"""
        n_1 = self.n_samples_per_candidate
        n_2 = self.n_samples_per_candidate
        # empirical means
        x_1 = np.mean(self.samples[c_1, :])
        x_2 = np.mean(self.samples[c_2, :])

        """
        professor formula:
        y_bar = (n_1 * x_1 + n_2 * x_2) / (n_1 + n_2)  # pooled empirical mean
        z_score = (x_1 - x_2) / math.sqrt(y_bar * (1 - y_bar) * (1 / n_1 + 1 / n_2))
        """
        # z-test, variances known (assumed to be 1 for the moment)
        var_1 = 1
        var_2 = 1
        z_score = (x_1 - x_2) / math.sqrt((var_1 / n_1) + (var_2 / n_2))

        if z_score >= self.z_value(1 - self.alpha):
            # H_0 (null hypothesis) is rejected
            return True
        else:
            # cannot reject null hypothesis
            return False

    @staticmethod
    def z_value(confidence):
        """Returns the z-value at a certain confidence"""
        return stats.norm.ppf(1 - (1 - confidence) / 2)


if __name__ == '__main__':
    probabilities = [0.99, 0.5, 0.55, 0.2]
    MP = [20, 25, 30, 35]
    print("Real values: {}".format([a * b for a, b in zip(probabilities, MP)]))
    env = Environment(probabilities)
    results = [0, 0, 0, 0]

    # experiments
    for experiment in range(1000):
        learner = SeqKTestLearner(num_of_candidates=4, marginal_profits=MP, environment=env, alpha=0.05)
        winner_candidate = learner.start()
        results[winner_candidate] += 1

    print("Total wins by candidate: {}".format(results))
