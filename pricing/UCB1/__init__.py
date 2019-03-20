from pricing.UCB1.UCB1_Learner import *
from pricing.UCB1.Environment import *
import matplotlib.pyplot as mpl
import numpy as np


rewards_perExp = []
rewards_perExp2 = []
probabilities = [0.3, 0.8, 0.2, 0.3, 0.4]
probabilities2 = [0.3, 0.8, 0.2, 0.3, 0.4, 0.3, 0.6, 0.5]
environment = Environment(probabilities)
environment2 = Environment(probabilities2)
bestCandidateMean = 0.8
timeHorizon = 10000
numberOfExperiments = 100

for i in range(numberOfExperiments):
    # the UCB1 learner, after being initialized, tries each arm once
    learner = UCB1_Learner(probabilities)
    learner2 = UCB1_Learner(probabilities2)
    realizations = environment.get_realizations()
    realizations2 = environment2.get_realizations()
    learner.init_ucb1_algorithm(realizations)
    learner2.init_ucb1_algorithm(realizations2)

    for j in range(timeHorizon):
        pulledArm = learner.pull_arm()
        reward = environment.get_reward(pulledArm)
        learner.update_rewards(pulledArm, reward)

        pulledArm = learner2.pull_arm()
        reward = environment2.get_reward(pulledArm)
        learner2.update_rewards(pulledArm, reward)

    # collection of the pulled arms means, for each experiment
    rewards_perExp.append(learner.pulledArmsReward)
    rewards_perExp2.append(learner2.pulledArmsReward)

mpl.plot(np.cumsum(np.mean(bestCandidateMean - np.array(rewards_perExp), axis=0)), "r")
mpl.plot(np.cumsum(np.mean(bestCandidateMean - np.array(rewards_perExp2), axis=0)), "b")
mpl.show()