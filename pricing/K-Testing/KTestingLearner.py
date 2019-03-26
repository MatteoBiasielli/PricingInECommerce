import numpy as np

from pricing.Environment import Environment


class KTestingLearner:
    # time horizon (exploration phase) -> n° of samples, accuracy
    def __init__(self, num_of_candidates, tot_n_of_samples, environment):
        self.num_of_candidates = num_of_candidates
        self.tot_n_of_samples = tot_n_of_samples
        self.environment = environment
        self.n_samples_per_candidate = int(tot_n_of_samples / num_of_candidates)
        # each row refers to the collected samples of one candidate
        self.samples = np.empty((num_of_candidates, self.n_samples_per_candidate), int)

    def start(self):
        # collect the samples
        for i in range(self.n_samples_per_candidate):
            realizations = self.environment.get_realizations()
            self.samples[:, i] = realizations
        # apply hypothesis test
        # TODO


if __name__ == '__main__':
    env = Environment([0.3, 0.8, 0.2, 0.1])
    learner = KTestingLearner(num_of_candidates=4, tot_n_of_samples=1000, environment=env)
    learner.start()
