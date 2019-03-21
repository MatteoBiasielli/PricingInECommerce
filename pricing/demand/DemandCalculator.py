from pricing.data.DataPreprocesser import DataPreprocesser as DP
import seaborn as sb
import numpy as np
import matplotlib.pyplot as mplt
from pricing.data.preprocessing_functions import *


class DemandCalculator:

    def __init__(self, data, prices_col='Max_Price', smoothing_window_size=10):
        self.data = data
        self.pr_col = prices_col
        self.smoothing_window_size = smoothing_window_size
        self.rev_cum, self.smooth_rev_cum, self.indexes = self.__calculate_reverse_cumulative_distribution()
        self.title = None

    def __calculate_reverse_cumulative_distribution(self):
        vals = self.data.data[self.pr_col].values
        num = int(max(vals) + 1)
        quantum = 1 / len(vals)
        reverse_cum = np.zeros(num + int((self.smoothing_window_size + 1) / 2))
        smooth_reverse_cum = np.zeros(num + int((self.smoothing_window_size + 1) / 2))
        indexes = [i for i in range(num + int((self.smoothing_window_size + 1) / 2))]
        for val in vals:
            reverse_cum[0:int(val)] += quantum
        smooth_reverse_cum[0] = 1
        for i in range(1, num):
            smooth_reverse_cum[i] = np.mean(reverse_cum[max(0, int(i - self.smoothing_window_size / 2)):
                                                        int(i + self.smoothing_window_size / 2)])

        return reverse_cum, smooth_reverse_cum, indexes

    def plot(self, smooth_reverse=True, original_reverse=False, show=True, title="Demand", legend_label="",
             scale=None, phases_labels=None):
        if scale is None:
            scale = [1]
            phases_labels = [""]
        for i in range(len(scale)):
            sc = scale[i]
            lab = legend_label if phases_labels is None else legend_label + " " + phases_labels[i]
            self.title = title
            if smooth_reverse:
                pic = sb.lineplot(self.indexes, self.smooth_rev_cum * sc, label=lab)
            if original_reverse:
                pic = sb.lineplot(self.indexes, self.rev_cum * sc, label="Raw data")
        if show:
            pic.set_title(self.title)
            pic.legend()
            pic.set_ylabel("Quantity")
            pic.set_xlabel("Price")
            mplt.show()
            self.title = None

    def get_demand_at(self, x):
        if int(x) == x:
            return self.smooth_rev_cum[int(x)]
        abs_diff = abs(int(x)-x)
        return self.smooth_rev_cum[int(x)] + (self.smooth_rev_cum[int(x) + 1] - self.smooth_rev_cum[int(x)])*abs_diff


if __name__ == '__main__':
    # AGGREGATED DEMAND
    all_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
    all_dem = DemandCalculator(all_people,
                               smoothing_window_size=50)
    all_dem.plot(smooth_reverse=True,
                 show=False,
                 legend_label="Aggregated",
                 # scale=[0.5, 0.7, 1],
                 phases_labels=['@Low Interest Phase', '@Med Interest Phase', '@High Interest Phase'])

    # DISAGGREGATED DEMANDS
    a_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
    a_people.process_data([
        select_where(column="Location_SouthEU", equals_to=1),
        select_where(column="Employed", equals_to=1)])
    a = DemandCalculator(a_people,
                              smoothing_window_size=50)
    a.plot(smooth_reverse=True,
           show=False,
           legend_label="South Europe = 1 and Employed = 1",
           # scale=[0.5, 0.7, 1],
           phases_labels=['@Low Interest Phase', '@Med Interest Phase', '@High Interest Phase'])

    b_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
    b_people.process_data([
        select_where(column="Location_SouthEU", equals_to=0)])
    b = DemandCalculator(b_people,
                            smoothing_window_size=50)
    b.plot(smooth_reverse=True,
           show=False,
           legend_label="South Europe = 0",
           #scale=[0.5, 0.7, 1],
           phases_labels=['@Low Interest Phase', '@Med Interest Phase', '@High Interest Phase'])

    c_people = DP(path='../data/preprocessed_data/processed_data.csv', no_basic_preprocessing=True)
    c_people.process_data([
        select_where(column="Location_SouthEU", equals_to=1),
        select_where(column="Employed", equals_to=0)])
    c = DemandCalculator(c_people,
                            smoothing_window_size=50)
    c.plot(smooth_reverse=True, legend_label="South Europe = 1 and Employed = 0",
           #scale=[0.5, 0.7, 1],
           phases_labels=['@Low Interest Phase', '@Med Interest Phase', '@High Interest Phase'])
