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


def plot_rewards(environment, marginal_profits, rewards, label, show=True):
    """"Plot the mean rewards of all the experiments against the clairvoyant one."""
    mean_reward = [sum(x) / len(rewards) for x in zip(*rewards)]
    if show:
        mpl.plot(mean_reward)
        print("Final value: {}".format(mean_reward[-1]))
        best_reward = np.max(np.array(environment.get_probabilities()) * np.array(marginal_profits))
        mpl.plot([best_reward] * len(mean_reward), "--k")
        mpl.legend(["{} ({} exps)".format(label, len(rewards)), "Clairvoyant Avg Reward"])
        # mpl.savefig("{}.png".format("rewards"), dpi=200)
        mpl.show()
    return mean_reward


def plot_cumulative_regret(environment, marginal_profits, rewards, show=True):
    """"Plot the cumulative regret averaging all the experiments."""
    mean_reward = [sum(x) / len(rewards) for x in zip(*rewards)]
    best_reward = np.max(np.array(environment.get_probabilities()) * np.array(marginal_profits))
    cum_regret = np.cumsum(best_reward - mean_reward)
    if show:
        mpl.plot(cum_regret, "r")
        mpl.legend(["Cumulative Regret ({} exps)".format(len(rewards))])
        # mpl.savefig("{}.png".format("cumregret"), dpi=200)
        mpl.show()
    return list(cum_regret)


def plot_cumulative_reward(rewards, show=True):
    """"Plot the cumulative reward averaging all the experiments."""
    mean_reward = [sum(x) / len(rewards) for x in zip(*rewards)]
    cum_reward = np.cumsum(mean_reward)
    if show:
        mpl.plot(cum_reward)
        mpl.legend(["Cumulative Reward ({} exps)".format(len(rewards))])
        mpl.show()
    return list(cum_reward)


def plot_multiple_curves(list_of_curves, title, labels, extend=False):
    """Plot multiple curves in the same figure."""
    if extend:
        # if the curves have different lengths, they are extended with the same slope
        max_len = max(len(c) for c in list_of_curves)
        for c in list_of_curves:
            if len(c) != max_len:
                diff = max_len - len(c)
                new_curve = c.copy()
                slope = new_curve[-1] - new_curve[-2]
                for i in range(diff):
                    new_curve.append((new_curve[-1] + slope))
                list_of_curves[list_of_curves.index(c)] = new_curve

    for curve in list_of_curves:
        mpl.plot(curve)
    mpl.legend(labels)
    mpl.title(title)
    mpl.savefig("{}.png".format(title), dpi=200)
    mpl.show()
