from pricing.demand.DemandCalculator import *
from pricing.UCB1.Environment import Environment
from pricing.kTesting.SeqKTestLearner import SeqKTestLearner
import matplotlib.pyplot as mpl

all_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
all_dem = DemandCalculator(all_people, smoothing_window_size=50)

probabilities = []
for i in [30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 290, 310, 330, 350]:
    probabilities.append(all_dem.get_demand_at(i))

MP = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340]
print("Real values: {}".format([a * b for a, b in zip(probabilities, MP)]))
env = Environment(probabilities)
results = [0] * len(probabilities)
history_of_rewards = []
history_of_mean_rewards = []
n_experiments = 100

# experiments
for experiment in range(n_experiments):
    learner = SeqKTestLearner(num_of_candidates=17, marginal_profits=MP, environment=env, alpha=0.05)
    winner_candidate = learner.start()
    results[winner_candidate] += 1
    history_of_rewards.append(learner.get_rewards_collected())
    history_of_mean_rewards.append(learner.get_mean_rewards_collected())

print("Total wins by candidate: {}".format(results))

# plot actual rewards collected
actual_reward = [sum(x) / len(history_of_rewards) for x in zip(*history_of_rewards)]
mpl.plot(actual_reward)
best_reward = np.max(np.array(env.get_probabilities()) * np.array(MP))
mpl.plot([best_reward] * len(actual_reward), "--k")
mpl.legend(["Actual Reward ({} exps)".format(n_experiments), "Clairvoyant Avg Reward"])
mpl.show()

# plot mean rewards during phases
mean_reward = [sum(x) / len(history_of_mean_rewards) for x in zip(*history_of_mean_rewards)]
mpl.plot(mean_reward)
mpl.plot([best_reward] * len(mean_reward), "--k")
mpl.legend(["Avg Reward ({} exps)".format(n_experiments), "Clairvoyant Avg Reward"])
mpl.show()
