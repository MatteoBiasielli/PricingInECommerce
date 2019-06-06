import csv
import pricing.ts.TSLearner as ts
import pricing.UCB1.Environment as env
from pricing import Environment as nsenv
import matplotlib.pyplot as plt
import numpy as np
import math
from pricing.kTesting.utility_functions import get_candidates
from pricing.demand.DemandCalculator import *

# --- DATA EXTRACTION --- #
data = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
dem = DemandCalculator(data, smoothing_window_size=50)

# ----- PARAMETERS ----- #

# Init parameters
n_arms = 4
candidate_prices = get_candidates(30, 350, n_arms)
marginal_cost = 10
time_horizon = 18250
n_experiments = 10000

sw_0 = int(math.sqrt(time_horizon))
sw_1 = int(n_arms*math.sqrt(time_horizon))
sw_2 = int(time_horizon / 3)
sw_3 = int(time_horizon / 6)
sw_4 = time_horizon
sw = sw_3

isNonStationary = True
isProfitMaximization = True
debug = False
storeResults = True


# Derived parameters
probabilities = []  # Structure is [] containing [] of probabilities

if isNonStationary:
    probabilities.append([dem.get_demand_at(i, scale="low") for i in candidate_prices])
    probabilities.append([dem.get_demand_at(i, scale="med") for i in candidate_prices])

probabilities.append([dem.get_demand_at(i, scale="high") for i in candidate_prices])
marginal_profits = [i-marginal_cost for i in candidate_prices]

if isProfitMaximization:
    if isNonStationary:
        profits = [[probabilities[j][i] * marginal_profits[i] for i in range(0, len(marginal_profits))]
                   for j in range(0, len(probabilities))]
    else:
        profits = [[probabilities[0][i] * marginal_profits[i] for i in range(0, len(marginal_profits))]]

rewards_per_experiment = []


# -- PREREQUISITES CHECK -- #

if len(probabilities[0]) != len(marginal_profits):
    print("ERROR: Length of arm probabilities and marginal profits do not match.")
    exit()
if time_horizon != int(time_horizon/len(probabilities))*len(probabilities):
    print("WARNING: Time horizon is not a multiple of the phase time."
          "The duration of the last phase will be slightly longer.")

# ------- ALGORITHM ------- #

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


# -- FORMATTING RESULTS -- #

n_phases = len(probabilities)
phase_time = int(time_horizon / n_phases)
excess = time_horizon - n_phases*phase_time

if isProfitMaximization:
    clairvoyant_reward = [np.full(phase_time, np.max(profits[i])) for i in range(n_phases)]
    clairvoyant_reward.append(np.full(excess, np.max(profits[n_phases-1])))
else:
    clairvoyant_reward = [np.full(phase_time, np.max(probabilities[i])) for i in range(n_phases)]
    clairvoyant_reward.append(np.full(excess, np.max(probabilities[n_phases-1])))

if isNonStationary:
    clairvoyant_reward = np.concatenate(clairvoyant_reward)
    clairvoyant_rewards_per_experiment = np.repeat(np.array([clairvoyant_reward]), n_experiments, axis=0)
else:
    clairvoyant_rewards_per_experiment = np.repeat(np.array(clairvoyant_reward), n_experiments, axis=0)
    clairvoyant_reward = np.squeeze(clairvoyant_reward)


# ---- STORING RESULTS ---- #
if storeResults:
    if isNonStationary:
        exp_rew = np.mean(rewards_per_experiment, axis=0)
        with open("results/ts_3phases" + str(sw) + "sw" + str(n_arms) + "arms_exprew_" + str(n_experiments) + "exp.csv",
                  "w") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(exp_rew)
        writeFile.close()

        clr_rew = clairvoyant_reward
        with open("results/ts_3phases" + str(sw) + "sw" + str(n_arms) + "arms_clrrew_" + str(n_experiments) + "exp.csv",
                  "w") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(clr_rew)
        writeFile.close()

        cum_reg = np.cumsum(np.mean(clairvoyant_rewards_per_experiment - rewards_per_experiment, axis=0))
        with open("results/ts_3phases" + str(sw) + "sw" + str(n_arms) + "arms_cumreg_" + str(n_experiments) + "exp.csv",
                  "w") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(cum_reg)
        writeFile.close()

    else:
        exp_rew = np.mean(rewards_per_experiment, axis=0)
        with open("results/ts_1phase" + str(n_arms) + "arms_exprew_" + str(n_experiments) + "exp.csv", "w") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(exp_rew)
        writeFile.close()

        clr_rew = clairvoyant_reward
        with open("results/ts_1phase" + str(n_arms) + "arms_clrrew_" + str(n_experiments) + "exp.csv", "w") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(clr_rew)
        writeFile.close()

        cum_reg = np.cumsum(np.mean(clairvoyant_rewards_per_experiment - rewards_per_experiment, axis=0))
        with open("results/ts_1phase" + str(n_arms) + "arms_cumreg_" + str(n_experiments) + "exp.csv", "w") as writeFile:
            writer = csv.writer(writeFile)
            writer.writerow(cum_reg)
        writeFile.close()


# ---- PLOTTING RESULTS ---- #
plt.figure(figsize=(8, 4))
plt.title("Average Expected Reward")
plt.xlabel("Number Of Interactions")
plt.plot(exp_rew)
plt.plot(clr_rew, "--k")
plt.legend(["TS Reward", "Clairvoyant Reward"])
plt.savefig("plots/ts_3phases" + str(sw) + "sw" + str(n_arms) + "arms_avgExpRew_" + str(n_experiments) + "exp.png")
plt.show()

plt.figure(figsize=(8, 4))
plt.title("Average Cumulative Regret")
plt.xlabel("Number Of Interactions")
plt.plot(cum_reg, "r")
plt.savefig("plots/ts_3phases" + str(sw) + "sw" + str(n_arms) + "arms_avgCumReg_" + str(n_experiments) + "exp.png")
plt.show()
