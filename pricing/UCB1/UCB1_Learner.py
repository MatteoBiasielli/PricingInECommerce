import numpy as np


class UCB1_Learner:

    def __init__(self, num_of_arms, sliding_window=0, marginal_profits=0):
        self.t = 0
        self.pulledArmsReward = []
        self.rewards_per_arm = [[] for i in range(num_of_arms)]
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
        self.empiricalMeans += realizations

        # If the sliding window is present, push the first realization of each arm into that arm buffer
        if self.sliding_window > 0:
            for i in range(self.num_of_arms):
                self.rewards_per_arm[i][0] = realizations[i]

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
                return np.argmax(self.empiricalMeans + np.sqrt((2 * np.log(self.t)) / self.numOfSamples))
            else:
                return np.argmax(self.marginal_profits * (self.empiricalMeans + np.sqrt((2 * np.log(self.t)) / self.numOfSamples)))

    # Update the empirical mean and the number of samples of each arm
    def update_rewards(self, pulled_arm, reward, exp_reward):

        # Saves the expected_reward (expected_reward * marginal_profit) given the arm pulled at the current round
        # This operation is required to print the average expected reward / regret
        if self.marginal_profits == 0:
            self.pulledArmsReward.append(exp_reward)
        else:
            self.pulledArmsReward.append(exp_reward * self.marginal_profits[pulled_arm])

        # If the sliding window is not present, the mean and the number of observed samples of the pulled arm
        # are updated
        if self.sliding_window == 0:
            self.numOfSamples[pulled_arm] += 1  # The number of observed samples can only increase
            # recursive formula to update the mean of a given pulledArm
            self.empiricalMeans[pulled_arm] = (self.numOfSamples[pulled_arm] - 1) / self.numOfSamples[pulled_arm] * \
                                              self.empiricalMeans[pulled_arm] + reward / self.numOfSamples[pulled_arm]
        else:
            # Updates for each arm in case of sliding window:
            for i in range(self.num_of_arms):
                # If we're considering the pulled arm:
                if i == pulled_arm:
                    if self.rewards_per_arm[i][-1] == -1:  # If the last elem of the buffer is a black space:
                        self.numOfSamples[i] += 1  # The number of samples increases, otherwise it remains constant

                    self.rewards_per_arm[i].insert(0, reward)  # Push into the pulled arm's buffer the new sample
                # If we're considering another arm
                else:
                    if self.rewards_per_arm[i][-1] != -1:  # If the last elem of the buffer IS NOT a black space:
                        self.numOfSamples[i] -= 1  # the number of samples reduces, otherwise it remains constant

                    self.rewards_per_arm[i].insert(0, -1)  # Push into the buffer a blank space (this arm was not pulled this round)

                # Pop of the last element of the buffer
                self.rewards_per_arm[i].__delitem__(-1)

                # Empirical mean update looking at the buffer of the arm we're considering
                if self.numOfSamples[i] == 0:
                    self.empiricalMeans[i] = 0.0
                else:
                    self.empiricalMeans[i] = self.rewards_per_arm[i].count(1) / (self.sliding_window - self.rewards_per_arm[i].count(-1))
