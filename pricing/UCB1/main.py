from pricing.UCB1.UCB1_Learner import *
from pricing.UCB1.Environment import *
from pricing.Environment import *
import matplotlib.pyplot as mpl
import numpy as np
from pricing.kTesting.utility_functions import *
from pricing.demand.DemandCalculator import *


rewards_perExp = []
rewards_perExp2 = []
rewards_perExp3 = []
probabilities = []
probabilities2 = []
probabilities3 = []
NS_probabilities = []

numofArms = 16

all_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
all_dem = DemandCalculator(all_people, smoothing_window_size=50)

marginal_profits = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
marginal_profits2 = [20, 56, 92, 128, 164, 200, 236, 272, 308, 340]
marginal_profits3 = [20, 127, 234, 340]

candidates = get_candidates(30, 350, numofArms)
marginal_profits4 = []
for i in candidates:
    marginal_profits4.append(i - 10)


for i in candidates:
    probabilities.append(all_dem.get_demand_at(i, 'low'))
NS_probabilities.append(probabilities)
for i in candidates:
    probabilities2.append(all_dem.get_demand_at(i, 'med'))
NS_probabilities.append(probabilities2)
for i in candidates:
    probabilities3.append(all_dem.get_demand_at(i, 'high'))
NS_probabilities.append(probabilities3)

# 67.89211229186635 17 arms

# mean for sw = 37.8146842966272  result = 4474
# mean for sw 10 = 37.72520804137658 result = 4463
# mean for sw 4 = 32.88042516048422  result = 3890

# for i in [30, 66, 102, 138, 174, 210, 246, 282, 318, 350]:
# probabilities2.append(all_dem.get_demand_at(i))

# for i in [30, 137, 244, 350]:
# probabilities3.append(all_dem.get_demand_at(i))
timeHorizon = 18250
numberOfExperiments = 0

c_arm = np.max(np.array(probabilities3) * marginal_profits4)


for i in range(numberOfExperiments):
    print(i)
    environment = Environment(probabilities3)
    # The ucb1 learner, after being initialized, tries every arm once

    learner = UCB1_Learner(num_of_arms=numofArms, marginal_profits=marginal_profits4)
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

# result_file = open("Ucb1_spezzata_regret.txt", "w")
# for i in range(3, 18):
#     file = open("Ucb1_" + str(i) + "_regret.txt", "r")
#     reward = file.read()
#     reward = reward.split(";")
#     reward = list(map(float, reward))
#     result_file.write(str(reward.__getitem__(-1)) + ";")
#     file.close()

# clairvoyant = np.zeros(18250)
# clairvoyant += c_arm
#
# reward = np.mean(rewards_perExp, axis=0)
# file = open("Ucb1_" + str(numofArms) + "_reward.txt", "w")
# file.write(str(reward[0]))
# for i in range(1, len(reward)):
#     file.write(";" + str(reward[i]))
# file.close()
#
# regret = np.cumsum(clairvoyant - np.array(reward))
# file = open("Ucb1_" + str(numofArms) + "_regret.txt", "w")
# file.write(str(regret[0]))
# for i in range(1, len(regret)):
#     file.write(";" + str(regret[i]))
# file.close()


# mpl.plot(reward)
# mpl.plot(c_arm, "--k")
# mpl.legend(["Ucb1 4 Arms SW", "Clairvoyant Algorithm"])
# mpl.xlabel("Number Of Interactions")
# mpl.ylabel("Average Expected Reward")
# mpl.show()

# file = open("Ucb1_10_reward.txt", "r")
# result4 = file.read()
# result4 = result4.split(";")
# result4 = list(map(float, result4))
# result4 = np.cumsum(result4)
# file.close()
#
# file = open("Ucb1_5_reward.txt", "r")
# result2 = file.read()
# result2 = result2.split(";")
# result2 = list(map(float, result2))
# result2 = np.cumsum(result2)
# file.close()
#
# result = np.array(result4) - result2




#
# file = open("Ucb1_10_reward_sw_paperfmla.txt", "r")
# result3 = file.read()
# result3 = result3.split(";")
# result3 = list(map(float, result3))
# file.close()
#
# file = open("Ucb1_10_reward_sw_phaselen.txt", "r")
# result1 = file.read()
# result1 = result1.split(";")
# result1 = list(map(float, result1))
# file.close()


# mpl.plot(np.array(result))
# mpl.plot(np.array(result3))
# mpl.plot(np.array(result1))
# mpl.plot(c_arm, "--k")
# mpl.legend(["SW = none", "SW2 = 1350", "SW3 = 4463", "SW4 = 6083"])
# mpl.xlabel("Number Of Interaction")
# mpl.ylabel("G(k)")
# mpl.axvline(x=2758, color="k")
# mpl.show()
