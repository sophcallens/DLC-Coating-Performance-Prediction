import numpy as np

def log10_ticks(ax, axis="x"):
    if axis == "x":
        data = ax.get_xlim()
        setter = ax.set_xticks
        labeler = ax.set_xticklabels
    else:
        data = ax.get_ylim()
        setter = ax.set_yticks
        labeler = ax.set_yticklabels

    ticks = []
    labels = []

    n_min = int(np.floor(data[0]))
    n_max = int(np.ceil(data[1]))

    for n in range(n_min, n_max):
        for k in range(1, 10):
            pos = n + np.log10(k)
            ticks.append(pos)
            if k == 1:
                labels.append(rf"$10^{{{n}}}$")
            else:
                labels.append("")   # pas d'étiquette

    setter(ticks)
    labeler(labels)
