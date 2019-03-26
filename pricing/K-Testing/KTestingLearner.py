import math

import numpy as np
import scipy.stats as stats

from pricing.Environment import Environment


class KTestingLearner:
    # TODO profit maximization
    # time horizon (exploration phase) -> n° of samples, accuracy
    def __init__(self, environment, num_of_candidates, tot_n_of_samples, alpha=0.05):
        self.environment = environment
        self.num_of_candidates = num_of_candidates
        self.tot_n_of_samples = tot_n_of_samples
        self.alpha = alpha
        self.n_samples_per_candidate = int(tot_n_of_samples / num_of_candidates)
        # each row refers to the collected samples of one candidate
        self.samples = np.empty((num_of_candidates, self.n_samples_per_candidate), int)
        self.empirical_means = []

    def start(self):
        """Starts the k-testing algorithm"""
        # collect the samples
        for i in range(self.n_samples_per_candidate):
            realizations = self.environment.get_realizations()
            self.samples[:, i] = realizations
        # compute empirical means
        for candidate in range(self.num_of_candidates):
            self.empirical_means.append(np.count_nonzero(self.samples[candidate, :]) / self.n_samples_per_candidate)
        # select the winner candidate
        winner = self.__select_winner()
        if winner is not False:
            print("Candidate {} is the winner!".format(winner))
            return winner
        else:
            print("A winner could not be found, try lowering confidence and/or increasing samples")

    def __select_winner(self):
        """Selects the winning candidate if possible, otherwise returns False"""
        # sort the candidates by empirical means
        empirical_ranking = sorted(range(len(self.empirical_means)), key=lambda k: self.empirical_means[k],
                                   reverse=True)
        # starting from most promising candidate, test against all other candidates
        for candidate in empirical_ranking:
            is_winner = self.__test_against_others(candidate)
            if is_winner:
                return candidate

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
        n = self.n_samples_per_candidate  # TODO maybe diversify num of samples per candidate (with probabilities)
        x_1 = self.empirical_means[c_1]
        x_2 = self.empirical_means[c_2]
        y_bar = (n * x_1 + n * x_2) / (n + n)  # pooled empirical mean

        test_statistic = (x_1 - x_2) / math.sqrt(y_bar * (1 - y_bar) * (1 / n + 1 / n))
        confidence = self.__get_z_value(1 - self.alpha)

        if test_statistic >= confidence:
            # H_0 (null hypothesis) is rejected
            return True
        else:
            # cannot reject null hypothesis
            return False

    def __compute_conf_interval(self, candidate):
        """Computes the binomial confidence interval with normal approximation"""
        p = self.empirical_means[candidate]  # sample mean
        z_value = self.__get_z_value(1 - self.alpha)
        lower_bound = p - (z_value * math.sqrt((p * (1 - p)) / self.n_samples_per_candidate))
        upper_bound = p + (z_value * math.sqrt((p * (1 - p)) / self.n_samples_per_candidate))
        return lower_bound, upper_bound

    def __get_z_value(self, confidence):
        """Returns the z-value at a certain confidence"""
        return stats.norm.ppf(1 - (1 - confidence) / 2)


if __name__ == '__main__':
    env = Environment([0.7, 0.8, 0.2, 0.1])
    learner = KTestingLearner(num_of_candidates=4, tot_n_of_samples=1000, environment=env, alpha=0.05)
    learner.start()
