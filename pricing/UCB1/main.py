from pricing.UCB1.UCB1_Learner import *
from pricing.UCB1.Environment import *
import matplotlib.pyplot as mpl
import numpy as np


rewards_perExp = []
probabilities = [0.3, 0.8, 0.2, 0.4]
# NS_probabilities = [[0.15, 0.1, 0.2, 0.35], [0.35, 0.7, 0.2, 0.35], [0.5, 0.7, 0.8, 0.15]]
timeHorizon = 10000
numberOfExperiments = 500
marginal_profits = [10, 5, 10, 15]
environment = Environment(probabilities)

bestCandidateExpReward = np.max(np.array(probabilities) * np.array(marginal_profits))

for i in range(numberOfExperiments):
    print("Experiment nr: " + str(i))
    environment = Environment(probabilities)

    # The ucb1 learner, after being initialized, tries every arm once
    learner = UCB1_Learner(num_of_arms=4, marginal_profits=marginal_profits)
    realizations = environment.get_realizations()
    learner.init_ucb1_algorithm(realizations)

    for j in range(timeHorizon):

        # pull-observe-update cycle
        pulledArm = learner.pull_arm()
        reward = environment.get_reward(pulledArm)
        learner.update_rewards(pulledArm, reward[0], reward[1])

    # At the end of each experiment the rewards are collected in the following list
    rewards_perExp.append(learner.pulledArmsReward)

# in this example we're showing the regret and the average expected reward of the learner working with the data
# defined above using an approach of "profit maximization"
clairvoyant_arm = np.zeros(timeHorizon)
clairvoyant_arm += bestCandidateExpReward
mpl.plot(np.mean(rewards_perExp, axis=0))
mpl.plot(clairvoyant_arm, "--k")
mpl.legend(["Avg_exp_reward", "Clairvoyant_reward"])
mpl.show()
mpl.plot(np.cumsum(np.mean(bestCandidateExpReward - np.array(rewards_perExp), axis=0)), "r")
mpl.legend(["Avg_regret"])
mpl.show()
