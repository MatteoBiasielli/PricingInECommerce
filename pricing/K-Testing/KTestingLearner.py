import math

import matplotlib.pyplot as mpl
import numpy as np
import scipy.stats as stats

from pricing.Environment import Environment


class KTestingLearner:

    def __init__(self, environment, num_of_candidates, tot_samples, marginal_profits=None, alpha=0.05,
                 plot_avg_reward=True, plot_avg_regret=True):
        self.environment = environment
        self.num_of_candidates = num_of_candidates
        self.tot_samples = tot_samples
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
        self.n_samples_per_candidate = int(tot_samples / num_of_candidates)
        # each row refers to the collected samples of one candidate
        self.samples = np.empty((num_of_candidates, self.n_samples_per_candidate), int)
        self.empirical_means = []

    def start(self):
        """Starts the k-testing algorithm"""
        # collect the samples (buy / not buy)
        for i in range(self.n_samples_per_candidate):
            realizations = self.environment.get_realizations()
            self.samples[:, i] = realizations

        for candidate in range(self.num_of_candidates):
            if self.profitMaximization:
                # turn mean conversion rate to mean profit
                self.samples[candidate, :] = self.samples[candidate, :] * self.marginal_profits[candidate]

            # save empirical means in list (conversion rates or mean profits depending on case)
            self.empirical_means.append(np.mean(self.samples[candidate, :]))

        # select the winner candidate
        winner = self.__select_winner()

        if not winner:
            return None
        else:
            if self.plot_avg_reward:
                self.__plot_avg_reward(winner)
            if self.plot_avg_regret:
                self.__plot_avg_regret(winner)
            return winner

    def __select_winner(self):
        """Selects the winning candidate if possible, otherwise returns False"""
        means = self.empirical_means.copy()
        if self.profitMaximization:
            # normalize mean profits
            means = (self.empirical_means - np.amin(self.empirical_means)) / (
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
        best_empirical = np.argmax(means)
        is_winner = self.__test_against_others(best_empirical, means)
        if is_winner:
            return best_empirical
        # if no candidate is a winner then return False
        return False

    def __test_against_others(self, candidate, means):
        """Tests if the given candidate has greater mean than all of the other candidates with statistical
        significance"""
        is_winner = True
        for other_candidate in [x for x in range(self.num_of_candidates) if x != candidate]:
            test = self.__hypothesis_test(candidate, other_candidate, means)
            # if even one test is inconclusive -> it's not a winner
            if test is False:
                is_winner = False
                break

        return is_winner

    def __hypothesis_test(self, c_1, c_2, means):
        """Performs an hypothesis test where the null hypothesis is: H_0 = u_c1 - u_c2 = 0 and the alternative one
        is: H_1 = u_c1 - u_c2 > 0"""
        n_1 = self.n_samples_per_candidate
        n_2 = self.n_samples_per_candidate
        x_1 = means[c_1]
        x_2 = means[c_2]

        y_bar = (n_1 * x_1 + n_2 * x_2) / (n_1 + n_2)  # pooled empirical mean
        z_score = (x_1 - x_2) / math.sqrt(y_bar * (1 - y_bar) * (1 / n_1 + 1 / n_2))
        p_value = stats.norm.sf(abs(z_score)) * 2  # two-tail

        if p_value <= self.alpha:
            # H_0 (null hypothesis) is rejected
            return True
        else:
            # cannot reject null hypothesis
            return False

    def __plot_avg_reward(self, winner):
        # plot best candidate reward
        if self.profitMaximization:
            best_reward = np.max(np.array(self.environment.get_probabilities()) * np.array(self.marginal_profits))
            winner_reward = self.environment.get_probabilities()[winner] * self.marginal_profits[winner]
        else:
            best_reward = self.environment.get_best_reward()
            winner_reward = self.environment.get_probabilities()[winner]
        clairvoyant_arm = np.zeros(2 * self.tot_samples)
        clairvoyant_arm += best_reward
        mpl.plot(clairvoyant_arm, "--k")
        # plot actual reward
        exploration = [np.mean(self.empirical_means)] * self.tot_samples
        exploitation = [winner_reward] * self.tot_samples
        actual = exploration + exploitation
        mpl.plot(actual)

        step = round((max(clairvoyant_arm) - min(actual)) / 10, 2)
        mpl.yticks(np.arange(round(min(actual), 2), max(clairvoyant_arm), step))
        mpl.legend(["Clairvoyant Avg Reward", "Actual Avg Reward"])
        mpl.show()

    def __plot_avg_regret(self, winner):
        if self.profitMaximization:
            best_reward = np.max(np.array(self.environment.get_probabilities()) * np.array(self.marginal_profits))
            winner_reward = self.environment.get_probabilities()[winner] * self.marginal_profits[winner]
        else:
            best_reward = self.environment.get_best_reward()
            winner_reward = self.environment.get_probabilities()[winner]
        exploration = [best_reward - np.mean(self.empirical_means)] * self.tot_samples
        exploitation = [best_reward - winner_reward] * self.tot_samples
        actual = exploration + exploitation
        mpl.plot(actual, "r")

        step = round(max(actual) / 10, 2)
        mpl.yticks(np.arange(0, round(max(actual), 2), step))
        mpl.legend(["Avg Regret"])
        mpl.show()


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
