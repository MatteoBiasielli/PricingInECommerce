import math
import queue
import random

import matplotlib.pyplot as mpl
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
        self.history_of_samples = []
        self.history_of_mean_rewards = []

    def start(self):
        """Starts the k-testing algorithm that returns one winner candidate at the end of the process"""
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

        # collect some winner samples
        self.__collect_samples(control)

        return control

    def __test_against(self, variation, control):
        """Tests the variation candidate against the control one, if the test is successful the variation is returned,
         otherwise the variation is discarded. Each time the samples are collected anew."""
        # for both collect samples
        self.__collect_samples(control)
        self.__collect_samples(variation)
        # shuffle last two set of samples collected (more realistic)
        last_samples = self.history_of_samples[-2 * self.n_samples_per_candidate:]
        random.shuffle(last_samples)
        self.history_of_samples[-2 * self.n_samples_per_candidate:] = last_samples
        # compute mean of true rewards in this phase
        last_true_rewards = self.history_of_mean_rewards[-2 * self.n_samples_per_candidate:]
        self.history_of_mean_rewards[-2 * self.n_samples_per_candidate:] = [np.mean(last_true_rewards)] * (
                2 * self.n_samples_per_candidate)
        # apply hypothesis test (test if mean of variation is higher than mean of control)
        if self.__hypothesis_test(variation, control) is True:
            # variation is better than control
            return variation
        else:
            # there is no evidence that variation is better than control
            return control

    def __collect_samples(self, candidate):
        """Collects a fixed number of samples for the specified candidate and updates the history of samples"""
        # collect the samples (buy / not buy)
        for i in range(self.n_samples_per_candidate):
            realization, true_value = self.environment.get_reward(candidate)
            if self.profitMaximization:
                realization = realization * self.marginal_profits[candidate]
                true_value = true_value * self.marginal_profits[candidate]
            self.samples[candidate, i] = realization
            self.history_of_samples.append(realization)
            self.history_of_mean_rewards.append(true_value)

    def __hypothesis_test(self, c_1, c_2):
        """Performs an hypothesis test where the null hypothesis is: H_0 = u_c1 - u_c2 = 0 and the alternative one
        is: H_1 = u_c1 - u_c2 > 0"""
        n_1 = self.n_samples_per_candidate
        n_2 = self.n_samples_per_candidate
        # empirical means
        x_1 = np.mean(self.samples[c_1, :])
        x_2 = np.mean(self.samples[c_2, :])

        if self.profitMaximization:
            # z-test, variances known (assumed to be 1 for the moment)
            var_1 = 1
            var_2 = 1
            z_score = (x_1 - x_2) / math.sqrt((var_1 / n_1) + (var_2 / n_2))

        else:
            # z-test on proportions
            y_bar = (n_1 * x_1 + n_2 * x_2) / (n_1 + n_2)  # pooled empirical mean
            z_score = (x_1 - x_2) / math.sqrt(y_bar * (1 - y_bar) * (1 / n_1 + 1 / n_2))

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

    def get_rewards_collected(self):
        """Returns the history of rewards collected during one experiment"""
        return self.history_of_samples.copy()

    def plot_reward(self):
        """Plots the actual rewards received during one experiment"""
        mpl.plot(self.history_of_samples)
        mpl.show()

    def get_mean_rewards_collected(self):
        """Plots the mean rewards during all phases of one experiment (exploration and exploitation)"""
        return self.history_of_mean_rewards.copy()


if __name__ == '__main__':
    probabilities = [0.8, 0.4, 0.5, 0.2]
    MP = [20, 25, 30, 35]
    print("Real values: {}".format([a * b for a, b in zip(probabilities, MP)]))
    env = Environment(probabilities)
    results = [0, 0, 0, 0]
    history_of_rewards = []
    history_of_mean_rewards = []
    n_experiments = 5000

    # experiments
    for experiment in range(n_experiments):
        learner = SeqKTestLearner(num_of_candidates=4, marginal_profits=MP, environment=env, alpha=0.05)
        winner_candidate = learner.start()
        results[winner_candidate] += 1
        history_of_rewards.append(learner.get_rewards_collected())
        history_of_mean_rewards.append(learner.get_mean_rewards_collected())

    print("Total wins by candidate: {}".format(results))

    # plot actual rewards collected
    actual_reward = [sum(x) / len(history_of_rewards) for x in zip(*history_of_rewards)]
    mpl.plot(actual_reward)
    best_reward = np.max(np.array(env.get_probabilities()) * np.array(MP))
    mpl.plot([best_reward] * len(actual_reward), "--k")
    mpl.legend(["Actual Reward ({} exps)".format(n_experiments), "Clairvoyant Avg Reward"])
    mpl.show()

    # plot mean rewards during phases
    mean_reward = [sum(x) / len(history_of_mean_rewards) for x in zip(*history_of_mean_rewards)]
    mpl.plot(mean_reward)
    mpl.plot([best_reward] * len(mean_reward), "--k")
    mpl.legend(["Avg Reward ({} exps)".format(n_experiments), "Clairvoyant Avg Reward"])
    mpl.show()
