from pricing.UCB1.UCB1_Learner import *
from pricing.UCB1.Environment import *
from pricing.Environment import *
import matplotlib.pyplot as mpl
import numpy as np
from pricing.demand.DemandCalculator import *


rewards_perExp = []
probabilities = []

all_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
all_dem = DemandCalculator(all_people, smoothing_window_size=50)

for i in [30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]:
    probabilities.append(all_dem.get_demand_at(i))

NS_probabilities = [[0.15, 0.1, 0.2, 0.35], [0.35, 0.7, 0.2, 0.45], [0.5, 0.7, 0.8, 0.15]]
timeHorizon = 18250
numberOfExperiments = 10000
marginal_profits = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
environment = Environment(probabilities)

bestCandidateExpReward = np.max(np.array(probabilities) * np.array(marginal_profits))

for i in range(numberOfExperiments):
    print("Experiment nr: " + str(i))
    environment = Environment(probabilities)

    # The ucb1 learner, after being initialized, tries every arm once
    learner = UCB1_Learner(num_of_arms=17, marginal_profits=marginal_profits)
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
mpl.title("Expected Reward")
mpl.xlabel("Number Of Interactions")
mpl.ylabel("Average_exp_reward")
mpl.legend(["UCB1_reward", "Clairvoyant_reward"])
mpl.show()
# mpl.plot(np.cumsum(np.mean(environment.clairvoyant_arm - np.array(rewards_perExp), axis=0)), "r")
# mpl.legend(["Avg_regret"])
# mpl.show()


