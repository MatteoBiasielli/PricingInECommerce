import numpy as np
from numba import jit
from tqdm import tqdm
import seaborn as sb
import matplotlib.pyplot as mplt
import copy
import utils as u
import pricing.demand.DemandCalculator as DC


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
    def get_best_arm(self, class_probs, cont_gen, arms, b=0.4):
        scores = np.zeros(len(arms))
        for i in range(len(arms)):
            for j in self.dem_ind_list:
                if cont_gen.n(i, j) == 0:
                    return i
        for i in range(len(arms)):
            for j in self.dem_ind_list:
                nij = cont_gen.n(i, j)
                scores[i] += (cont_gen.emp_mean(i, j) +
                              b * np.sqrt(2 * np.log10(max(min(min(cont_gen.time_steps, cont_gen.sw_size)
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
        return score - np.sqrt(-np.log10(0.1) * 2 / (cum_prob * n_cum))

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
        self.current_scale_factor = 1
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
                sort = np.random.binomial(n=1, size=1, p=self.arm_demand_means[bestarm][dem]*self.current_scale_factor)[0]
                self.realizations_per_arm_per_demand[bestarm][dem].append(sort)
                self.realizations_per_arm_per_demand_timestamps[bestarm][dem].append(self.time_steps)
            # print(self.realizations_per_arm_per_demand)

            rew += node.get_arm_reward(self.class_probabilities, self, bestarm, self.arms)

        self.__clean_old_samples()

        rew = min(rew, clair)
        self.clairvoyants.append(clair)
        self.rewards.append(rew)
        self.regrets.append(clair - rew)

    def __clean_old_samples(self):
        try:
            if self.sw_size > 0:
                found = False
                while True:
                    for i in range(self.arm_demand_means.shape[0]):
                        for j in range(self.arm_demand_means.shape[1]):
                            if self.realizations_per_arm_per_demand_timestamps[i][j][0] < self.time_steps - self.sw_size:
                                self.realizations_per_arm_per_demand[i][j].pop(0)
                                self.realizations_per_arm_per_demand_timestamps[i][j].pop(0)
                                found = True
                    if not found:
                        break
                    found = False
        except IndexError:
            return

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
    n_arms_max = 17
    for n_arms in [5, 10, 17]:
        sw_size = 3000
        prod_cost = 10
        arms = np.array([30 + (320/(n_arms-1)) * i for i in range(n_arms)])
        print(arms)
        class_probs = np.array([108/329, 48/329, 173/329])
        arm_dem_means = np.zeros([3, len(arms), len(class_probs)])
        a, b, c, = DC.get_three_demands()
        phases = ['low', 'med', 'high']
        dems = [a, b, c]
        for ph in range(len(phases)):
            for i in range(len(arms)):
                for j in range(len(class_probs)):
                    arm_dem_means[ph][i][j] = dems[j].get_demand_at(arms[i], scale=phases[ph])
        print(arm_dem_means)
        clairs = []
        rews = []
        regrs = []
        T_HOR = 18250
        N_EXPS = 30
        arms = arms - prod_cost
        for _ in tqdm(range(N_EXPS)):
            ph = 0
            contgen = ContextTreeGeneratorUCB1(class_probs,
                                               arm_dem_means[ph],
                                               arms, sw_size,
                                               ["SEU_E", "NSEU", "SEU_NE"])
            for i in range(T_HOR):
                contgen.calc_clair_rew_regret()
                if i % 350 == 0:
                    contgen.split()
                if (i + 1) % 6084 == 0:
                    contgen.arm_demand_means = arm_dem_means[ph+1]
                    ph += 1
            clairs.append(contgen.clairvoyants)
            rews.append(contgen.rewards)
            regrs.append(contgen.regrets)
            contgen.print_tree()
        # sb.lineplot(range(T_HOR), u.smoothen_curve(np.mean(clairs, axis=0)))
        smooth_rews = u.smoothen_curve(np.mean(rews, axis=0))
        gr = sb.lineplot(range(T_HOR), smooth_rews)
        gr.set_title(str(n_arms) + " ARMS - REVENUE")
        mplt.show()
        smooth_regrs = u.smoothen_curve(np.mean(regrs, axis=0))
        gr = sb.lineplot(range(T_HOR), smooth_regrs)
        gr.set_title(str(n_arms) + " ARMS - REGRET")
        mplt.show()
        clairmean = np.mean(clairs, axis=0)
        gr = sb.lineplot(range(T_HOR), clairmean)
        gr.set_title(str(n_arms) + " ARMS - CLAIRVOYANT")
        mplt.show()
        u.save_mat_file("./saves_ucb1/sw3000_nonstat_" + str(n_arms) + ".mat", {"regrets": smooth_regrs,
                                                                      "rewards": smooth_rews,
                                                                      "clairs": clairmean})


















