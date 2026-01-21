"""Operating parameters plot

Create 3 graphs in fonction of the ratio CoF / Wear rate

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from log_scale import log10_ticks

# =========================
# Inputs
# =========================
inputs = [
    {'name': 'log10(Sliding velocity (m/s))', 'log': True},
    {'name': 'log10(Sliding distance (m))', 'log': True},
    {'name': 'Load (N)', 'log': False}
]

ratio_label = 'log10(Friction coefficient / Wear rate)'

# =========================
# Load & clean data
# =========================
df = pd.read_csv(
    "data/cleaned_dataset.csv",
    sep=";",
    decimal=",",
    encoding="utf-8"
)

df = df.dropna(subset=[
    *(i['name'] for i in inputs),
    'Friction coefficient',
    'log10(Wear rate)'
])

# =========================
# Ratio (Y axis)
# =========================
df['ratio'] = np.log10(
    df['Friction coefficient'] /
    10**df['log10(Wear rate)']
)

# =========================
# Figure + gridspec
# =========================
fig = plt.figure(figsize=(18,5))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05])

axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax = fig.add_subplot(gs[0, 3])  # colorbar axis

# =========================
# Plots
# =========================
for ax, inp in zip(axes, inputs):

    x = df[inp['name']].values
    y = df['ratio'].values

    hb = ax.hexbin(
        x, y,
        gridsize=20,
        cmap='viridis',
        mincnt=1,
        linewidths=0.1,
        edgecolors='none'
    )

    ax.set_xlabel(inp['name'])
    ax.grid(alpha=0.3)

    if inp['log']:
        log10_ticks(ax, axis="x")

    # Y axis log ticks (shared)
    log10_ticks(ax, axis="y")

axes[0].set_ylabel(ratio_label)

# =========================
# Colorbar (properly placed)
# =========================
cbar = fig.colorbar(hb, cax=cax)
cbar.set_label('Number of points')

# =========================
# Title & layout
# =========================
fig.suptitle(
    'Density of friction / wear ratio vs operating parameters',
    fontsize=14
)

plt.tight_layout()


# Save


out_path = f'make_figures/figures/Density of friction / wear ratio vs operating parameters.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

