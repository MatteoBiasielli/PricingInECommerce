import numpy as np


class UCB1_Learner:

    def __init__(self, num_of_arms, sliding_window=0):
        self.t = 0
        self.pulledArmsReward = []
        self.rewards_per_arm = [[] for i in range(num_of_arms)]
        self.empiricalMeans = np.zeros(num_of_arms)
        self.numOfSamples = np.zeros(num_of_arms)
        self.num_of_arms = num_of_arms
        self.sliding_window = sliding_window
        if self.sliding_window > 0:
            for i in range(num_of_arms):
                for k in range(sliding_window):
                    self.rewards_per_arm[i].append(-1)

    # given a sample for each arm, initializes the empirical means and the number of samples, furthermore, the
    # time istant becomes now equal to 1
    def init_ucb1_algorithm(self, realizations):
        self.t += 1
        self.numOfSamples += 1
        self.empiricalMeans += realizations

        if self.sliding_window > 0:
            for i in range(self.num_of_arms):
                self.rewards_per_arm[i][0] = realizations[i]

    # chooses the best arm according to ucb1 criteria, Increments the number of samples of the chosen arm
    # then saves the mean of the Bernoulli associated at that pulled arm
    def pull_arm(self):
        print("means: " + str(self.empiricalMeans))
        print("samples: " + str(self.numOfSamples))
        self.t += 1
        if self.numOfSamples.__contains__(0):
            for i in range(len(self.numOfSamples)):
                if self.numOfSamples[i] == 0:
                    return i
        else:
            return np.argmax(self.empiricalMeans + np.sqrt((2 * np.log(self.t)) / self.numOfSamples))

    # exploits a recursive formula to update the empirical mean of the chosen arm
    def update_rewards(self, pulled_arm, reward, exp_reward):
        # mean saving
        self.pulledArmsReward.append(exp_reward)

        if self.sliding_window == 0:
            self.numOfSamples[pulled_arm] += 1
            # recursive formula to update the mean of a given pulledArm
            self.empiricalMeans[pulled_arm] = (self.numOfSamples[pulled_arm] - 1) / self.numOfSamples[pulled_arm] * \
                                              self.empiricalMeans[pulled_arm] + reward / self.numOfSamples[pulled_arm]
        else:
            for i in range(self.num_of_arms):
                # update of the numbers of sample
                if i == pulled_arm:
                    if self.rewards_per_arm[i][-1] == -1:
                        self.numOfSamples[i] += 1

                    self.rewards_per_arm[i].insert(0, reward)
                else:
                    if self.rewards_per_arm[i][-1] != -1:
                        self.numOfSamples[i] -= 1

                    self.rewards_per_arm[i].insert(0, -1)

                # pop of the last element of the buffer
                self.rewards_per_arm[i].__delitem__(-1)

                if self.numOfSamples[i] == 0:
                    self.empiricalMeans[i] = 0.0
                else:
                    self.empiricalMeans[i] = self.rewards_per_arm[i].count(1) / (self.sliding_window - self.rewards_per_arm[i].count(-1))


