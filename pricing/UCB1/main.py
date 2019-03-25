from pricing.UCB1.UCB1_Learner import *
from pricing.UCB1.Environment import *
import matplotlib.pyplot as mpl
from pricing.UCB1.NonStationaryEnvironment import *
import numpy as np


rewards_perExp = []
rewards_perExp2 = []
probabilities = [0.3, 0.8, 0.2, 0.3, 0.4]
probabilities2 = [[0.15, 0.1, 0.2, 0.35], [0.35, 0.7, 0.2, 0.35], [0.5, 0.1, 0.8, 0.15]]
timeHorizon = 7000
numberOfExperiments = 100
environment = Environment(probabilities)

bestCandidateMean = np.max(probabilities)

for i in range(numberOfExperiments):
    print(i)
    # the UCB1 learner, after being initialized, tries each arm once
    learner = UCB1_Learner(4, 1000)
    learner2 = UCB1_Learner(4)
    env = NonStationaryEnvironment(probabilities2, timeHorizon)
    env2 = NonStationaryEnvironment(probabilities2, timeHorizon)
    realizations = env.get_realizations()
    learner.init_ucb1_algorithm(realizations)
    realizations = env2.get_realizations()

    learner2.init_ucb1_algorithm(realizations)

    for j in range(timeHorizon):
        pulledArm = learner.pull_arm()
        pulledArm2 = learner2.pull_arm()
        # print("pull" + str(pulledArm))
        reward = env.get_reward(pulledArm)
        reward2 = env2.get_reward(pulledArm2)
        learner.update_rewards(pulledArm, reward[0], reward[1])
        learner2.update_rewards(pulledArm2, reward2[0], reward2[1])

    # collection of the pulled arms means, for each experiment
    rewards_perExp.append(learner.pulledArmsReward)
    rewards_perExp2.append(learner2.pulledArmsReward)


mpl.plot(np.cumsum(np.mean(np.array(env.clairvoyant_arm) - np.array(rewards_perExp), axis=0)), "b")
mpl.plot(np.cumsum(np.mean(np.array(env2.clairvoyant_arm) - np.array(rewards_perExp2), axis=0)), "r")
mpl.legend(["sw_ucb1", "ucb1"])
# mpl.plot(np.mean(rewards_perExp, axis=0), "r")
# mpl.plot(np.mean(rewards_perExp2, axis=0))
# mpl.plot(env2.clairvoyant_arm, "--k")
mpl.show()


