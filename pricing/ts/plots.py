import matplotlib.pyplot as plt
from numpy import genfromtxt, arange, subtract, cumsum, repeat

PATH = "./results/"
KTPATH = "../kTesting/results/"
UCB1PATH = "../UCB1/Results/"


def get_data_from_csv(filename, path=PATH, delimiter=","):
    return genfromtxt(path + filename, delimiter=delimiter)


def plot_1phase_metricsOverInteractions(toplot="rew", pfrom=3, upto=17, save=False):
    plt.figure(figsize=(16, 8))
    xlabel = "Number Of Interactions"
    ylabel = "Average Expected Reward" if toplot == "rew" else "Average Cumulative Regret"
    load_subname = "exprew" if toplot == "rew" else "cumreg"
    save_subname = "avgExpRew" if toplot == "rew" else "avgCumReg"
    arms_subname = "allArms" if pfrom == 3 and upto == 17 else str(pfrom) + "to" + str(upto)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    legend = []
    cm = plt.get_cmap("hsv")
    num_colours = 8

    for i in range(pfrom, upto + 1):
        data = get_data_from_csv("ts_1phase" + str(i) + "arms_" + load_subname + "_10000exp.csv")
        line_style = "solid" if ((i-3) % 2) == 0 else "dashed"
        line_colour = cm((int((i-pfrom)/2))/num_colours)
        plt.plot(data, linestyle=line_style, color=line_colour)

        legend.append(str(i) + " arms")

    plt.legend(legend)

    if save:
        plt.savefig("plots/ts_1phase_" + save_subname + "_" + arms_subname + ".png")

    plt.show()


def plot_1phase_finalMetricsOverArms(toplot="rew", pfrom=3, upto=17, save=False):
    plt.figure(figsize=(16, 8))
    xlabel = "Number Of Arms"
    ylabel = "Final Average Expected Reward" if toplot == "rew" else " Final Average Cumulative Regret"
    load_subname = "exprew" if toplot == "rew" else "cumreg"
    save_subname = "finExpRew" if toplot == "rew" else "finCumReg"
    arms_subname = "allArms" if pfrom == 3 and upto == 17 else str(pfrom) + "to" + str(upto)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    legend = ["TS Reward", "Clairvoyant Reward"] if toplot == "rew" else ["TS Regret"]

    conv_data = []
    clr_data = []
    arms_data = []

    for i in range(pfrom, upto + 1):
        data = get_data_from_csv("ts_1phase" + str(i) + "arms_" + load_subname + "_10000exp.csv")
        conv_data.append(data[len(data)-1])

        if toplot == "rew":
            data = get_data_from_csv("ts_1phase" + str(i) + "arms_clrrew_10000exp.csv")
            clr_data.append(data[len(data)-1])
            legend.append("Clairvoyant Reward")

        arms_data.append(i)

    plt.plot(arms_data, conv_data)
    if toplot == "rew":
        plt.plot(arms_data, clr_data, "--k")

    plt.xticks(arms_data)
    plt.legend(legend)

    if save:
        plt.savefig("plots/ts_1phase_" + save_subname + "_" + arms_subname + ".png")

    plt.show()


def plot_3phases_metricsOverInteractions(toplot="rew", n_arms=17, save=False):
    plt.figure(figsize=(16, 8))
    xlabel = "Number Of Interactions"
    ylabel = "Average Expected Reward" if toplot == "rew" else "Average Cumulative Regret"
    load_subname = "exprew" if toplot == "rew" else "cumreg"
    save_subname = "avgExpRew" if toplot == "rew" else "avgCumReg"
    arms_subname = str(n_arms)
    second_sw = 540 if n_arms == 4 else 1350 if n_arms == 10 else 2296
    sw = [135, second_sw, 3041, 6083, 18250]
    plt.title(str(n_arms) + " Arms")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    legend = []

    for w in sw:
        data = get_data_from_csv("ts_3phases" + str(w) + "sw" + arms_subname + "arms_" + load_subname + "_10000exp.csv")
        plt.plot(data)
        legend.append("Sliding Window = " + str(w))

        if toplot == "rew" and w == 18250:
            data = get_data_from_csv("ts_3phases" + str(w) + "sw" + arms_subname + "arms_clrrew_10000exp.csv")
            plt.plot(data, "--k")
            legend.append("Clairvoyant Reward")

    plt.legend(legend)

    if save:
        plt.savefig("plots/ts_3phases_" + save_subname + "_" + arms_subname + ".png")

    plt.show()


def plot_3phases_gfunOverInteractions(sw=3041, save=False):
    plt.figure(figsize=(16, 8))
    xlabel = "Number Of Interactions"
    ylabel = "G-Function"
    plt.title("Sliding Window = " + str(sw))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    legend = ["G(4, 10)", "G(4, 17)"]

    ref_data = get_data_from_csv("ts_3phases" + str(sw) + "sw4arms_exprew_10000exp.csv")
    data_10 = get_data_from_csv("ts_3phases" + str(sw) + "sw10arms_exprew_10000exp.csv")
    data_17 = get_data_from_csv("ts_3phases" + str(sw) + "sw17arms_exprew_10000exp.csv")

    gfun_10 = subtract(cumsum(data_10), cumsum(ref_data))
    gfun_17 = subtract(cumsum(data_17), cumsum(ref_data))

    plt.plot(gfun_10)
    plt.plot(gfun_17)
    plt.plot(repeat(0, 18250), "--k")

    plt.legend(legend)

    if save:
        plt.savefig("plots/ts_3phases_gFun_" + str(sw) + ".png")

    plt.show()


def plot_1phase_finalMetricsOverArms_Comparison(toplot="rew", pfrom=3, upto=17, save=False):
    plt.figure(figsize=(16, 8))
    xlabel = "Number Of Arms"
    ylabel = "Final Average Expected Reward" if toplot == "rew" else " Final Average Cumulative Regret"
    load_subname = "exprew" if toplot == "rew" else "cumreg"
    dondony_subname = "reward" if toplot == "rew" else "regret"
    save_subname = "finExpRew" if toplot == "rew" else "finCumReg"
    arms_subname = "allArms" if pfrom == 3 and upto == 17 else str(pfrom) + "to" + str(upto)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    legend = ["K-Testing Reward", "UCB1 Reward", "TS Reward", "Clairvoyant Reward"] if toplot == "rew" \
        else["K-Testing Regret", "UCB1 Regret", "TS Regret"]

    conv_data_kt = []
    conv_data_ucb1 = []
    conv_data_ts = []
    clr_data = []
    arms_data = []

    data_ucb1 = get_data_from_csv("Ucb1_spezzata_" + dondony_subname + ".txt", path=UCB1PATH, delimiter=";")
    for i in range(pfrom, upto + 1):
        data_kt = get_data_from_csv(load_subname + "_" + str(i) + ".csv", path=KTPATH)
        data_ts = get_data_from_csv("ts_1phase" + str(i) + "arms_" + load_subname + "_10000exp.csv")

        conv_data_kt.append(data_kt[len(data_kt) - 1])
        conv_data_ts.append(data_ts[len(data_ts) - 1])

        conv_data_ucb1.append(data_ucb1[i - 3])

        if toplot == "rew":
            data = get_data_from_csv("ts_1phase" + str(i) + "arms_clrrew_10000exp.csv")
            clr_data.append(data[len(data) - 1])

        arms_data.append(i)

    plt.plot(arms_data, conv_data_kt)
    plt.plot(arms_data, conv_data_ucb1)
    plt.plot(arms_data, conv_data_ts)

    if toplot == "rew":
        plt.plot(arms_data, clr_data, "--k")

    plt.xticks(arms_data)
    plt.legend(legend)

    if save:
        plt.savefig("plots/ts_1phase_" + save_subname + "_" + arms_subname + ".png")

    plt.show()


plot_1phase_finalMetricsOverArms_Comparison(toplot="reg", pfrom=4)
