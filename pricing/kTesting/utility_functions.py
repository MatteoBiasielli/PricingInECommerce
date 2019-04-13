import matplotlib.pyplot as mpl
import numpy as np


def get_candidates(start, end, n_candidates):
    """Get equally separated candidates given a range of prices and a number of candidates specified."""
    step = (end - start) / (n_candidates - 1)
    candidates = [start]
    for x in range(n_candidates - 2):
        candidates.append(round(candidates[-1] + step))
    candidates.append(end)
    return candidates


def plot_rewards(environment, marginal_profits, rewards, label):
    """"Plot the mean rewards of all the experiments against the clairvoyant one."""
    mean_reward = [sum(x) / len(rewards) for x in zip(*rewards)]
    mpl.plot(mean_reward)
    best_reward = np.max(np.array(environment.get_probabilities()) * np.array(marginal_profits))
    mpl.plot([best_reward] * len(mean_reward), "--k")
    mpl.legend(["{} ({} exps)".format(label, len(rewards)), "Clairvoyant Avg Reward"])
    mpl.show()
