from pricing.UCB1.Environment import Environment
from pricing.demand.DemandCalculator import *
from pricing.kTesting.SeqKTestLearner import SeqKTestLearner
from pricing.kTesting.utility_functions import get_candidates, plot_rewards

# get demand curve:
all_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
all_dem = DemandCalculator(all_people, smoothing_window_size=50)

# select candidates and cost to get probabilities and marginal profits:
candidates = get_candidates(start=30, end=350, n_candidates=17)
probabilities = [all_dem.get_demand_at(c) for c in candidates]
cost = 10
marginal_profits = [x - cost for x in candidates]

# setup experiments:
print("Real values: {}".format([a * b for a, b in zip(probabilities, marginal_profits)]))
env = Environment(probabilities)
results = [0] * len(probabilities)
history_of_actual_rewards = []
history_of_mean_rewards = []
n_experiments = 1000

# experiments:
for experiment in range(n_experiments):
    learner = SeqKTestLearner(num_of_candidates=len(probabilities), marginal_profits=marginal_profits, environment=env,
                              n_samples=2000)
    winner_candidate = learner.start()
    results[winner_candidate] += 1
    history_of_actual_rewards.append(learner.get_rewards_collected())
    history_of_mean_rewards.append(learner.get_mean_rewards_collected())

# visualize results:
print("Total wins by candidate: {}".format(results))
plot_rewards(env, marginal_profits, history_of_mean_rewards, "Avg Reward")
