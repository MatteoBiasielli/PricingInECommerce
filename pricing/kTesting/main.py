from pricing.UCB1.Environment import Environment
from pricing.demand.DemandCalculator import *
from pricing.kTesting.SeqKTestLearner import SeqKTestLearner
from pricing.kTesting.utility_functions import get_candidates, plot_rewards, plot_cumulative_regret, \
    plot_multiple_curves


def run_experiments(demand_curve, n_candidates):
    # select candidates and cost to get probabilities and marginal profits:
    cost = 10
    candidates = get_candidates(start=30, end=350, n_candidates=n_candidates)
    probabilities = [demand_curve.get_demand_at(c) for c in candidates]
    marginal_profits = [x - cost for x in candidates]

    # setup experiments:
    tot_datapoints = 18250
    n_samples_each = round((tot_datapoints / (n_candidates - 1)) / 2)
    print("With {} candidates:".format(n_candidates))
    print("\tReal values: {}".format([a * b for a, b in zip(probabilities, marginal_profits)]))
    env = Environment(probabilities)
    results = [0] * len(probabilities)
    history_of_actual_rewards = []
    history_of_mean_rewards = []

    # experiments:
    n_experiments = 100
    for experiment in range(n_experiments):
        learner = SeqKTestLearner(num_of_candidates=len(probabilities), marginal_profits=marginal_profits,
                                  environment=env, n_samples=n_samples_each)
        winner_candidate = learner.start()
        results[winner_candidate] += 1
        history_of_actual_rewards.append(learner.get_rewards_collected())
        history_of_mean_rewards.append(learner.get_mean_rewards_collected())

    print("\tTotal wins by candidate: {}".format(results))
    return plot_rewards(env, marginal_profits, history_of_mean_rewards, "Avg Reward",
                        show=False), plot_cumulative_regret(env, marginal_profits, history_of_mean_rewards, show=False)


if __name__ == '__main__':
    # get demand curve:
    all_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
    all_dem = DemandCalculator(all_people, smoothing_window_size=50)

    # test different n of candidates:
    rewards_to_plot = []
    regret_to_plot = []
    candidates_to_test = [4, 8, 10, 17]
    for n in candidates_to_test:
        results = run_experiments(all_dem, n)
        rewards_to_plot.append(results[0])
        regret_to_plot.append(results[1])

    # visualize results:
    plot_multiple_curves(rewards_to_plot, candidates_to_test)
    plot_multiple_curves(regret_to_plot, candidates_to_test)
