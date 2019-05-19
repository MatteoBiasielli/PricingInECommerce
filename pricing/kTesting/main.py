from pricing.UCB1.Environment import Environment
from pricing.demand.DemandCalculator import *
from pricing.kTesting.SeqKTestLearner import SeqKTestLearner
import math
from pricing.kTesting.utility_functions import get_candidates, plot_rewards, plot_cumulative_regret, \
    plot_multiple_curves, plot_cumulative_reward


def run_experiments(demand_curve, n_candidates, alpha, phase="high"):
    # select candidates and cost to get probabilities and marginal profits:
    cost = 10
    candidates = get_candidates(start=30, end=350, n_candidates=n_candidates)
    probabilities = [demand_curve.get_demand_at(c, scale=phase) for c in candidates]
    marginal_profits = [x - cost for x in candidates]

    # setup experiments:
    tot_datapoints = 18250
    n_samples_each = math.floor((tot_datapoints / (n_candidates - 1)) / 2)
    print("With {} candidates, {} alpha, {} phase:".format(n_candidates, alpha, phase))
    print("\tReal values: {}".format([round(a * b, 3) for a, b in zip(probabilities, marginal_profits)]))
    env = Environment(probabilities)
    wins = [0] * len(probabilities)
    history_of_mean_rewards = []

    # experiments:
    n_experiments = 10000
    for experiment in range(n_experiments):
        learner = SeqKTestLearner(num_of_candidates=len(probabilities), marginal_profits=marginal_profits,
                                  environment=env, n_samples=n_samples_each, alpha=alpha)
        winner_candidate = learner.start()
        wins[winner_candidate] += 1
        history_of_mean_rewards.append(learner.get_mean_rewards_collected())

    print("\tTotal wins by candidate: {}".format(wins))
    return plot_rewards(env, marginal_profits, history_of_mean_rewards, "Avg Reward",
                        show=True), plot_cumulative_regret(env, marginal_profits, history_of_mean_rewards,
                                                           show=True)


def save_results(list_of_values, filename):
    with open(filename, 'w') as f:
        f.write("%f" % list_of_values[0])
        list_of_values.pop(0)
        for value in list_of_values:
            f.write(",%f" % value)


if __name__ == '__main__':
    # get demand curve:
    all_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
    all_dem = DemandCalculator(all_people, smoothing_window_size=50)

    # test different candidates/alpha:
    rewards_to_plot = []
    regret_to_plot = []
    candidates_to_test = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    alpha_to_test = []
    for i in candidates_to_test:
        results = run_experiments(all_dem, n_candidates=i, alpha=0.3, phase="high")
        save_results(results[0], "exprew_{}.csv".format(i))
        save_results(results[1], "cumreg_{}.csv".format(i))
        # rewards_to_plot.append(results[0])
        # regret_to_plot.append(results[1])

    # visualize results:
    # plot_multiple_curves(rewards_to_plot, title="Expected Reward", labels=alpha_to_test, extend=True)
    # plot_multiple_curves(regret_to_plot, title="Cumulative Regret", labels=candidates_to_test, extend=True)
