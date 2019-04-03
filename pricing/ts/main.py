import pricing.ts.TSLearner as ts
import pricing.UCB1.Environment as env
from pricing import Environment as nsenv
import matplotlib.pyplot as plt
import numpy as np


rewards_per_experiment = []
probabilities = [0.5, 0.3, 0.7]
ns_probabilities = [[0.5, 0.3, 0.7], [0.85, 0.3, 0.3]]
time_horizon = 10000
n_experiments = 1000
marginal_profits = 0
sw = 1000

isNonStationary = True
debug = False

for exp in range(n_experiments):
    print("Experiment " + str(exp))

    if isNonStationary:
        environment = nsenv.NonStationaryEnvironment(ns_probabilities, time_horizon)
        learner = ts.TSLearner(n_arms=len(ns_probabilities[0]), sliding_window=sw)
    else:
        environment = env.Environment(probabilities)
        learner = ts.TSLearner(n_arms=len(probabilities))

    realizations = environment.get_realizations()

    for i in range(time_horizon):
        arm_to_pull = learner.get_best_arm_from_prior()
        reward = environment.get_reward(arm_to_pull)
        learner.update(arm_to_pull, reward[0], reward[1])

        if debug:
            print("Time: " + str(learner.t) + " Arm: " + str(arm_to_pull) + " Reward: " + str(reward))

    rewards_per_experiment.append(learner.expected_rewards)

if isNonStationary:
    """phase_time = int(time_horizon / len(ns_probabilities))
    clairvoyant_reward = [np.full(phase_time, np.max(ns_probabilities[i])) for i in range(len(ns_probabilities))]
    clairvoyant_reward = np.array(clairvoyant_reward)
    clairvoyant_reward = np.ndarray.flatten(clairvoyant_reward)
    clairvoyant_reward = np.repeat(np.array([clairvoyant_reward]), n_experiments, axis=0)
    """
    clairvoyant_reward = np.full(int(time_horizon/2), 0.7)
    clairvoyant_reward = np.concatenate((clairvoyant_reward, np.full(int(time_horizon/2), 0.85)))


else:
    clairvoyant_reward = np.full(time_horizon, np.max(probabilities))

plt.figure(figsize=(8, 4))
plt.plot(np.mean(rewards_per_experiment, axis=0))
plt.plot(clairvoyant_reward, "--k")
plt.legend(["Avg_exp_reward", "Clairvoyant_reward"])
plt.show()
plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(np.mean(clairvoyant_reward - rewards_per_experiment, axis=0)), "r")
plt.legend(["Avg_regret"])
plt.show()
