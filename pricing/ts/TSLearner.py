import numpy as np
from collections import deque


class TSLearner:

    def __init__(self, n_arms, sliding_window=0, marginal_profits=[]):
        self.n_arms = n_arms
        self.sliding_window = sliding_window
        self.marginal_profits = marginal_profits

        self.t = 0
        self.reward_per_arm = [[] for i in range(n_arms)]
        self.expected_rewards = []
        self.beta_params = np.ones((n_arms, 2))
        self.isProfitMaximization = (len(marginal_profits) > 0)
        self.isSlidingWindow = (sliding_window > 0)

        if self.isSlidingWindow:
            self.buffer = deque(np.repeat([[0, 0]], sliding_window, axis=0), maxlen=sliding_window)  # pairs (arm, realization)

    # Returns the index of the arm which corresponds to the highest sampled value from the related prior
    def get_best_arm_from_prior(self):
        thetas = np.random.beta(self.beta_params[:, 0], self.beta_params[:, 1])

        if self.isProfitMaximization:
            thetas = thetas * self.marginal_profits

        best_arm = np.argmax(thetas)
        return best_arm

    def update(self, pulled_arm, reward, exp_reward):
        self.t += 1

        if self.isProfitMaximization:
            self.reward_per_arm[pulled_arm].append(reward*self.marginal_profits[pulled_arm])
            self.expected_rewards.append(exp_reward*self.marginal_profits[pulled_arm])
        else:
            self.reward_per_arm[pulled_arm].append(reward)
            self.expected_rewards.append(exp_reward)

        self.beta_params[pulled_arm][0] += reward      # alpha
        self.beta_params[pulled_arm][1] += 1 - reward  # beta

        if self.isSlidingWindow:
            self.buffer.append((pulled_arm, reward))
            oldest_sample = self.buffer.popleft()

            if self.t > self.sliding_window:
                oldest_arm = oldest_sample[0]
                oldest_realization = oldest_sample[1]

                self.beta_params[oldest_arm][0] = max(1, self.beta_params[oldest_arm][0] - oldest_realization)
                self.beta_params[oldest_arm][1] = max(1, self.beta_params[oldest_arm][1] - (1 - oldest_realization))



