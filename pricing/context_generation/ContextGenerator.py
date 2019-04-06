import numpy as np
from numba import jit
from tqdm import tqdm
import seaborn as sb
import matplotlib.pyplot as mplt
import copy


class Node:

    def __init__(self, demands_index_list):
        self.dem_ind_list = demands_index_list

    @jit
    def get_clairvoyant(self, class_probs, arm_demand_means, arms):
        scores = np.zeros(len(arms))
        for i in range(arm_demand_means.shape[0]):
            for j in self.dem_ind_list:
                scores[i] += arm_demand_means[i][j] * arms[i] * class_probs[j]
        return np.max(scores)

    @jit
    def get_best_arm(self, class_probs, cont_gen, arms):
        scores = np.zeros(len(arms))
        for i in range(len(arms)):
            for j in self.dem_ind_list:
                if cont_gen.n(i, j) == 0:
                    return i
        for i in range(len(arms)):
            for j in self.dem_ind_list:
                nij = cont_gen.n(i, j)
                scores[i] += (cont_gen.emp_mean(i, j) +
                              np.sqrt(2 * np.log10(max(min(min(cont_gen.time_steps, cont_gen.sw_size)
                                                   if cont_gen.sw_size > 0 else cont_gen.time_steps, nij), 2)) / nij)) \
                             * arms[i] * class_probs[j]
        # print(self.dem_ind_list, scores)
        return np.argmax(scores)

    @jit
    def get_arm_reward(self, class_probs, cont_gen, arm, arms):
        score = 0
        for j in self.dem_ind_list:
            score += cont_gen.emp_mean(arm, j) * arms[arm] * class_probs[j]
        return score

    @jit
    def get_arm_hoeffding_lowbound(self, class_probs, cont_gen, arm, arms):
        score = 0
        cum_prob = 0
        n_cum = 0
        for j in self.dem_ind_list:
            score += (cont_gen.emp_mean(arm, j)) \
                     * arms[arm] * class_probs[j]
            cum_prob += class_probs[j]
            n_cum += cont_gen.n(arm, j)
        # print(arm, n_cum)
        return score - np.sqrt(-np.log10(0.025) * 2 / (cum_prob * n_cum))

    def split(self, class_probs, cont_gen, arms):
        if len(self.dem_ind_list) <= 1:
            raise CantSplitException()
        for i in range(len(arms)):
            for dem in self.dem_ind_list:
                if cont_gen.n(i, dem) == 0:
                    raise CantSplitException()
        for i in range(len(self.dem_ind_list)):
            spl1 = [self.dem_ind_list[i]]
            spl2 = copy.deepcopy(self.dem_ind_list)
            spl2.pop(i)
            n1 = Node(spl1)
            n2 = Node(spl2)
            b1 = n1.get_best_arm(class_probs, cont_gen, arms)
            b2 = n2.get_best_arm(class_probs, cont_gen, arms)
            b = self.get_best_arm(class_probs, cont_gen, arms)
            l1 = n1.get_arm_hoeffding_lowbound(class_probs, cont_gen, b1, arms)
            l2 = n2.get_arm_hoeffding_lowbound(class_probs, cont_gen, b2, arms)
            l = self.get_arm_hoeffding_lowbound(class_probs, cont_gen, b, arms)
            print(l1, l2, l)
            if l1 + l2 > l:
                return n1, n2
        raise CantSplitException()


class CantSplitException(Exception):
    def __init__(self):
        pass


class ContextTreeGeneratorUCB1:

    def __init__(self, class_probabilities, arm_demand_means, arms, sliding_window_size, demands_names=None):
        self.realizations_per_arm_per_demand = [[
            [] for _ in range(len(class_probabilities))
        ] for _ in range(len(arms))]
        self.realizations_per_arm_per_demand_timestamps = [[
            [] for _ in range(len(class_probabilities))
        ] for _ in range(len(arms))]
        self.time_steps = 0
        self.regrets = []
        self.clairvoyants = []
        self.rewards = []
        self.arms = arms
        self.sw_size = sliding_window_size
        self.class_probabilities = class_probabilities
        self.arm_demand_means = arm_demand_means
        self.demands_names = demands_names
        self.nodes = [Node([i for i in range(len(class_probabilities))])]

    @jit
    def n(self, arm, demand):
        val = min(len(self.realizations_per_arm_per_demand[arm][demand]), self.sw_size) if self.sw_size > 0 else \
                                                             len(self.realizations_per_arm_per_demand[arm][demand])
        # print(arm, demand, val)
        return val

    @jit
    def emp_mean(self, arm, dem):
        return np.mean(self.realizations_per_arm_per_demand[arm][dem][
                       len(self.realizations_per_arm_per_demand[arm][dem]) -
                       min(len(self.realizations_per_arm_per_demand[arm][dem]), self.sw_size):]) if self.sw_size > 0 \
                else np.mean(self.realizations_per_arm_per_demand[arm][dem])

    @jit
    def calc_clair_rew_regret(self):
        clair = 0
        rew = 0
        self.time_steps += 1
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            clair += node.get_clairvoyant(self.class_probabilities, self.arm_demand_means, self.arms)
            bestarm = node.get_best_arm(self.class_probabilities, self, self.arms)
            for dem in node.dem_ind_list:
                sort = np.random.binomial(n=1, size=1, p=self.arm_demand_means[bestarm][dem])[0]
                self.realizations_per_arm_per_demand[bestarm][dem].append(sort)
                self.realizations_per_arm_per_demand_timestamps[bestarm][dem].append(self.time_steps)
            # print(self.realizations_per_arm_per_demand)

            rew += node.get_arm_reward(self.class_probabilities, self, bestarm, self.arms)

        self.__clean_old_samples()

        rew = min(rew, clair)
        self.clairvoyants.append(clair)
        self.rewards.append(rew)
        self.regrets.append(clair - rew)

    @jit
    def __clean_old_samples(self):
        if self.sw_size > 0:
            found = False
            while not True:
                for i in range(self.arm_demand_means.shape[0]):
                    for j in range(self.arm_demand_means.shape[1]):
                        if self.realizations_per_arm_per_demand_timestamps[i][j][0] < self.time_steps - self.sw_size:
                            self.realizations_per_arm_per_demand[i][j].pop(0)
                            found = True
                if not found:
                    break
                found = False

    def split(self):
        for i in range(len(self.nodes)):
            try:
                n1, n2 = self.nodes[i].split(self.class_probabilities, self, self.arms)
                self.nodes.pop(i)
                self.nodes.append(n1)
                self.nodes.append(n2)
                break
            except CantSplitException:
                continue

    def print_tree(self):
        for node in self.nodes:
            linetopr = "IF "
            for dem in node.dem_ind_list:
                linetopr += self.demands_names[dem]
                if node.dem_ind_list[-1] != dem:
                    linetopr += " OR "
                else:
                    linetopr += " ---> "
            ba = node.get_best_arm(self.class_probabilities, self, self.arms)
            linetopr += "ARM " + str(ba)
            print(linetopr)


if __name__ == '__main__':
    class_probs = np.array([0.3, 0.5, 0.2])
    arm_dem_means = np.array([[0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.01, 0.4, 1]])
    arms = np.array([10, 9, 5])
    sw_size = 4000

    clairs = []
    rews = []
    regrs = []
    T_HOR = 10000
    N_EXPS = 6
    for _ in tqdm(range(N_EXPS)):
        contgen = ContextTreeGeneratorUCB1(class_probs, arm_dem_means, arms, sw_size, ["a", "b", "c"])
        for i in range(T_HOR):
            contgen.calc_clair_rew_regret()
            if i % 2000 == 0:
                contgen.split()
        clairs.append(contgen.clairvoyants)
        rews.append(contgen.rewards)
        regrs.append(contgen.regrets)
        contgen.print_tree()
    sb.lineplot(range(T_HOR), np.mean(clairs, axis=0))
    sb.lineplot(range(T_HOR), np.mean(rews, axis=0))
    mplt.show()
    sb.lineplot(range(T_HOR), np.mean(regrs, axis=0))
    mplt.show()


















