import utils as u
import os
import seaborn as sb
import matplotlib.pyplot as mplt
import numpy as np
"""
dirpaths = ["./saves_ts", "./saves_ucb1"]
filename = "sw3000_nonstat_"
max_n_arms = 17
toplot = [5]
sw = 3000
labels = ["TS", "UCB1"]
regrs = []
clairs = []
rews = []

for dirpath in dirpaths:
    for i in toplot:
        dictionary = u.read_mat_file(os.path.join(dirpath, filename + str(i) + ".mat"))
        print(dictionary)
        regrs.append(dictionary['regrets'][0])
        clairs.append(np.mean(dictionary['clairs'], axis=0))
        rews.append(dictionary['rewards'][0])

time = [i for i in range(len(regrs[0]))]

for j in range(len(toplot)):
    for i in range(j*(len(dirpaths)), (j+1)*(len(dirpaths))):
        sb.lineplot(x=time, y=rews[i], label="Reward " + labels[i])
        ax = sb.lineplot(x=time, y=clairs[i], label="Clairvoyant " + labels[i])
        ax.set_title(labels[i] + " with Context Generation - " + str(toplot[j]) + " Arms")
        ax.set_xlabel("Number of Observations")
        ax.set_ylabel("Average Reward")
    mplt.show()
"""

filename = ["./saves_ts/sw3000_nonstat_5.mat", "./saves_ts/sw4500_nonstat_5.mat", "./saves_ucb1/sw3000_nonstat_5.mat"]
regrs = []
clairs = []
rews = []

for fn in filename:
    dictionary = u.read_mat_file(fn)
    regrs.append(dictionary['regrets'][0])
    clairs.append(np.mean(dictionary['clairs'], axis=0))
    rews.append(dictionary['rewards'][0])

time = [i for i in range(len(regrs[0]))]
sb.lineplot(x=time, y=np.minimum(rews[0], clairs[2]), label="Reward TS - SW 3000")
sb.lineplot(x=time, y=np.minimum(rews[1], clairs[2]), label="Reward TS - SW 4500")
sb.lineplot(x=time, y=np.minimum(rews[2], clairs[2]), label="Reward UCB1 - SW 3000")
ax = sb.lineplot(x=time, y=clairs[2], label="Clairvoyant")
ax.set_title("TS and UCB1 with Context Generation - 5 Arms")
ax.set_xlabel("Number of Observations")
ax.set_ylabel("Average Reward")
mplt.show()
"""
filename = ["./saves_ucb1/sw3000_nonstat_17.mat"]
regrs = []
clairs = []
rews = []

for fn in filename:
    dictionary = u.read_mat_file(fn)
    regrs.append(dictionary['regrets'][0])
    clairs.append(np.mean(dictionary['clairs'], axis=0))
    rews.append(dictionary['rewards'][0])

time = [i for i in range(len(regrs[0]))]
sb.lineplot(x=time, y=np.minimum(rews[0], clairs[0]), label="Reward")
ax = sb.lineplot(x=time, y=clairs[0], label="Clairvoyant")
ax.set_title("UCB1 with Context Generation - 17 Arms")
ax.set_xlabel("Number of Observations")
ax.set_ylabel("Average Reward")
mplt.show()"""