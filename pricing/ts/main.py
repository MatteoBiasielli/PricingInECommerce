import pricing.ts.TSLearner as ts
import pricing.UCB1.Environment as env
from pricing import Environment as nsenv
import matplotlib.pyplot as plt
import numpy as np


# - PARAMETERS - #

# Init parameters
probabilities = np.array([[0.5, 0.3, 0.7], [0.5, 0.3, 0.95]]) # Structure is [] containing [] of probabilities
marginal_profits = [10, 8, 6]
time_horizon = 10000
n_experiments = 100
sw = 1000

isNonStationary = True
isProfitMaximization = True
debug = False

rewards_per_experiment = []

# Derived parameters
if isProfitMaximization:
    if isNonStationary:
        profits = probabilities * marginal_profits
    else:
        profits = probabilities[0] * marginal_profits


# - PREREQUISITES CHECK - #

if len(probabilities[0]) != len(marginal_profits):
    print("ERROR: Length of arm probabilities and marginal profits do not match.")
    exit()


# - ALGORITHM - #

for exp in range(n_experiments):
    print("Experiment " + str(exp))

    if isNonStationary:
        environment = nsenv.NonStationaryEnvironment(probabilities, time_horizon)
        learner = ts.TSLearner(n_arms=len(probabilities[0]), sliding_window=sw, marginal_profits=marginal_profits)
    else:
        environment = env.Environment(probabilities[0])
        learner = ts.TSLearner(n_arms=len(probabilities[0]), marginal_profits=marginal_profits)

    realizations = environment.get_realizations()

    for i in range(time_horizon):
        arm_to_pull = learner.get_best_arm_from_prior()
        reward = environment.get_reward(arm_to_pull)
        learner.update(arm_to_pull, reward[0], reward[1])

        if debug:
            print("Time: " + str(learner.t) + " Arm: " + str(arm_to_pull) + " Reward: " + str(reward))

    rewards_per_experiment.append(learner.expected_rewards)


# - FORMATTING RESULTS - #

n_phases = len(probabilities)
phase_time = int(time_horizon / n_phases)

if isProfitMaximization:
    clairvoyant_reward = [np.full(phase_time, np.max(profits[i])) for i in range(len(probabilities))]
else:
    clairvoyant_reward = [np.full(phase_time, np.max(probabilities[i])) for i in range(len(probabilities))]

if isNonStationary:
    clairvoyant_reward = np.concatenate(clairvoyant_reward)
    clairvoyant_rewards_per_experiment = np.repeat(np.array([clairvoyant_reward]), n_experiments, axis=0)
else:
    clairvoyant_rewards_per_experiment = np.repeat(np.array(clairvoyant_reward), n_experiments, axis=0)
    clairvoyant_reward = np.squeeze(clairvoyant_reward)

# - PLOTTING RESULTS - #
print(clairvoyant_reward)
plt.figure(figsize=(8, 4))
plt.plot(np.mean(rewards_per_experiment, axis=0))
plt.plot(clairvoyant_reward, "--k")
plt.legend(["Avg_exp_reward", "Clairvoyant_reward"])
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(np.cumsum(np.mean(clairvoyant_rewards_per_experiment - rewards_per_experiment, axis=0)), "r")
plt.legend(["Avg_regret"])
plt.show()
