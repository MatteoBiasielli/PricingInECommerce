import numpy as np


class UCB1_Learner:

    def __init__(self, num_of_arms, sliding_window=0, marginal_profits=0):
        self.t = 0
        self.pulledArmsReward = []
        self.rewards_per_arm = [[] for i in range(num_of_arms)]
        self.results = []
        self.ones = np.zeros(num_of_arms)
        self.zeros = np.zeros(num_of_arms)
        self.resultsSize = 0
        self.empiricalMeans = np.zeros(num_of_arms)
        self.numOfSamples = np.zeros(num_of_arms)
        self.num_of_arms = num_of_arms
        self.marginal_profits = marginal_profits
        self.sliding_window = sliding_window
        if self.sliding_window > 0:
            for i in range(num_of_arms):
                for k in range(sliding_window):
                    self.rewards_per_arm[i].append(-1)

    # Given a realization for each arm, initializes the empirical means and the number of samples, furthermore, the
    # time step becomes now equal to 1
    def init_ucb1_algorithm(self, realizations):
        self.t += 1
        self.numOfSamples += 1
        self.empiricalMeans += realizations  # TODO

        # If the sliding window is present, push the first realization of each arm into that arm buffer
        for i in range(len(realizations)):
            self.results.append([i, realizations[i]])
            self.resultsSize += 1
            if realizations[i] == 0:
                self.zeros[i] += 1
            else:
                self.ones[i] += 1

    # chooses the best arm according to ucb1 criteria, Increments the number of samples of the chosen arm
    # then saves the mean of the Bernoulli associated at that pulled arm
    def pull_arm(self):
        self.t += 1

        # This if_statement may be true only in presence of a sliding window
        if self.numOfSamples.__contains__(0):
            for i in range(len(self.numOfSamples)):
                if self.numOfSamples[i] == 0:
                    return i
        else:
            # Classic selection of the best arm in case of "market invasion" approach (marginal_profits = 0)
            # Alternative selection in case of "profit maximization" approach (marginal_profits given)
            if self.marginal_profits == 0:
                return np.argmax(self.empiricalMeans + 0.3 * np.sqrt((2 * np.log(self.t)) / self.numOfSamples))
            else:
                return np.argmax(self.marginal_profits * (self.empiricalMeans + 0.3 * np.sqrt((2 * np.log(self.t)) / self.numOfSamples)))

    # Update the empirical mean and the number of samples of each arm
    def update_rewards(self, pulled_arm, reward, exp_reward):

        # Saves the expected_reward (expected_reward * marginal_profit) given the arm pulled at the current round
        # This operation is required to print the average expected reward / regret
        if self.marginal_profits == 0:
            self.pulledArmsReward.append(exp_reward)
        else:
            self.pulledArmsReward.append(exp_reward * self.marginal_profits[pulled_arm])

        if reward == 0:
            self.zeros[pulled_arm] += 1
        else:
            self.ones[pulled_arm] += 1

        # If the sliding window is not present, the mean and the number of observed samples of the pulled arm
        # are updated
        if self.sliding_window == 0:
            self.numOfSamples[pulled_arm] += 1  # The number of observed samples can only increase
            # recursive formula to update the mean of a given pulledArm
            self.empiricalMeans[pulled_arm] = (self.numOfSamples[pulled_arm] - 1) / self.numOfSamples[pulled_arm] * \
                                              self.empiricalMeans[pulled_arm] + reward / self.numOfSamples[pulled_arm]
        else:
            if self.resultsSize == self.sliding_window:
                target = self.results[-1][0]
                value = self.results[-1][1]

                self.numOfSamples[target] -= 1

                if value == 0:
                    self.zeros[target] -= 1
                else:
                    self.ones[target] -= 1

                if self.ones[target] + self.zeros[target] > 0:
                    self.empiricalMeans[target] = self.ones[target] / (self.zeros[target] + self.ones[target])
                else:
                    self.empiricalMeans[target] = 0.0

                self.results.__delitem__(-1)

                self.results.insert(0, [pulled_arm, reward])

                if reward == 0:
                    self.zeros[pulled_arm] += 1
                else:
                    self.ones[pulled_arm] += 1

                self.numOfSamples[pulled_arm] += 1

                self.empiricalMeans[pulled_arm] = self.ones[pulled_arm] / (self.zeros[pulled_arm] + self.ones[pulled_arm])
            else:
                self.results.insert(0, [pulled_arm, reward])
                self.resultsSize += 1

                if reward == 0:
                    self.zeros[pulled_arm] += 1
                else:
                    self.ones[pulled_arm] += 1

                self.numOfSamples[pulled_arm] += 1

                self.empiricalMeans[pulled_arm] = self.ones[pulled_arm] / (self.zeros[pulled_arm] + self.ones[pulled_arm])
