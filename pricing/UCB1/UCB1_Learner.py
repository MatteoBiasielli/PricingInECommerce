import numpy as np


class UCB1_Learner:

    def __init__(self, probabilities):
        self.t = 0
        self.pulledArmsReward = []
        self.empiricalMeans = np.zeros(len(probabilities))
        self.numOfSamples = np.zeros(len(probabilities))
        self.probabilities = probabilities

    # given a sample for each arm, initializes the empirical means and the number of samples, furthermore, the
    # time istant becomes now equal to 1
    def init_ucb1_algorithm(self, realizations):
        self.t += 1
        self.numOfSamples += 1
        self.empiricalMeans += realizations

    # chooses the best arm according to ucb1 criteria, Increments the number of samples of the chosen arm
    # then saves the mean of the Bernulli associated at that pulled arm
    def pull_arm(self):
        self.t += 1
        tmp = np.argmax(self.empiricalMeans + np.sqrt((2 * np.log(self.t)) / self.numOfSamples))
        self.numOfSamples[tmp] += 1
        # mean saving
        self.pulledArmsReward.append(self.probabilities[tmp])
        return tmp

    # exploits a recursive formula to update the empirical mean of the chosen arm
    def update_rewards(self, pulled_arm, reward):
        # recursive formula to update the mean of a given pulledArm
        self.empiricalMeans[pulled_arm] = (self.numOfSamples[pulled_arm] - 1) / self.numOfSamples[pulled_arm] * self.empiricalMeans[pulled_arm] + reward / self.numOfSamples[pulled_arm]
