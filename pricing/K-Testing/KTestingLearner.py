import math

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as mpl

from pricing.Environment import Environment


class KTestingLearner:

    def __init__(self, environment, num_of_candidates, marginal_profits, tot_samples, alpha=0.05, plot_reward=True,
                 plot_regret=True):
        self.environment = environment
        self.num_of_candidates = num_of_candidates
        self.marginal_profits = marginal_profits
        self.tot_samples = tot_samples
        self.alpha = alpha
        self.plot_reward = plot_reward
        self.plot_regret = plot_regret
        self.n_samples_per_candidate = int(tot_samples / num_of_candidates)
        # each row refers to the collected samples of one candidate
        self.samples = np.empty((num_of_candidates, self.n_samples_per_candidate), int)
        self.empirical_means = []
        self.normalized_means = []

    def start(self):
        """Starts the k-testing algorithm"""
        # collect the samples
        for i in range(self.n_samples_per_candidate):
            realizations = self.environment.get_realizations()
            self.samples[:, i] = realizations

        for candidate in range(self.num_of_candidates):
            # compute empirical means
            mean_conversion_rate = np.count_nonzero(self.samples[candidate, :]) / self.n_samples_per_candidate
            self.empirical_means.append(mean_conversion_rate * self.marginal_profits[candidate])
            # turn mean conversion rate to mean profit
            self.samples[candidate, :] = self.samples[candidate, :] * self.marginal_profits[candidate]

        # select the winner candidate
        winner = self.__select_winner()

        if not winner:
            return None
        else:
            if self.plot_reward:
                self.__plot_reward(winner)
            return winner

    def __select_winner(self):
        """Selects the winning candidate if possible, otherwise returns False"""
        # normalize mean profits
        self.normalized_means = (self.empirical_means - np.amin(self.empirical_means)) / (
                np.amax(self.empirical_means) - np.amin(self.empirical_means))

        """
        # sort the candidates by empirical mean profits
        empirical_ranking = sorted(range(len(self.empirical_means)), key=lambda k: self.empirical_means[k],
                                   reverse=True)
        # starting from most promising candidate, test against all other candidates
        for candidate in empirical_ranking:
            is_winner = self.__test_against_others(candidate)
            if is_winner:
                return candidate
        """
        # test most promising candidate against all others
        best_empirical = self.normalized_means.argmax()
        is_winner = self.__test_against_others(best_empirical)
        if is_winner:
            return best_empirical
        # if no candidate is a winner then return False
        return False

    def __test_against_others(self, candidate):
        """Tests if the given candidate has greater mean than all of the other candidates with statistical
        significance"""
        is_winner = True
        for other_candidate in [x for x in range(self.num_of_candidates) if x != candidate]:
            test = self.__hypothesis_test(candidate, other_candidate)
            # if even one test is inconclusive -> it's not a winner
            if test is False:
                is_winner = False
                break

        return is_winner

    def __hypothesis_test(self, c_1, c_2):
        """Performs an hypothesis test where the null hypothesis is: H_0 = u_c1 - u_c2 = 0 and the alternative one
        is: H_1 = u_c1 - u_c2 > 0"""
        n_1 = self.n_samples_per_candidate
        n_2 = self.n_samples_per_candidate
        x_1 = self.normalized_means[c_1]
        x_2 = self.normalized_means[c_2]

        y_bar = (n_1 * x_1 + n_2 * x_2) / (n_1 + n_2)  # pooled empirical mean
        z_score = (x_1 - x_2) / math.sqrt(y_bar * (1 - y_bar) * (1 / n_1 + 1 / n_2))
        p_value = stats.norm.sf(abs(z_score)) * 2  # two-tail

        if p_value <= self.alpha:
            # H_0 (null hypothesis) is rejected
            return True
        else:
            # cannot reject null hypothesis
            return False

    def __plot_reward(self, winner):
        # plot best candidate reward
        best_reward = self.environment.get_best_profit_reward(self.marginal_profits)
        clairvoyant_arm = np.zeros(2 * self.tot_samples)
        clairvoyant_arm += best_reward
        mpl.plot(clairvoyant_arm, "--k")
        # plot actual reward
        exploration = [np.mean(self.empirical_means)] * self.tot_samples
        exploitation = [self.environment.get_probabilities()[winner] * self.marginal_profits[winner]] * self.tot_samples
        actual = exploration + exploitation
        mpl.plot(actual)

        mpl.legend(["Clairvoyant Avg Reward", "Actual Avg Reward"])
        mpl.show()

    def __plot_regret(self, winner):
        pass  # TODO


if __name__ == '__main__':
    probabilities = [0.6, 0.95, 0.85, 0.7]
    MP = [25, 30, 35, 40]
    print("Real values: {}".format([a * b for a, b in zip(probabilities, MP)]))
    env = Environment(probabilities)
    results = [0, 0, 0, 0]
    wnf = 0

    # experiments
    for experiment in range(1):
        learner = KTestingLearner(num_of_candidates=4, marginal_profits=MP, tot_samples=500,
                                  environment=env, alpha=0.05)
        winner_candidate = learner.start()
        if winner_candidate is not None:
            results[winner_candidate] += 1
        else:
            wnf += 1

    print("Total wins by candidate: {}".format(results))
    print("Total winner not found scenarios: {}".format(wnf))
