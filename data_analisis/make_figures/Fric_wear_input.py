"""Friction and Wear plots in fonction of input

Produce side-by-side bar plots of mean Wear and CoF for an input, with errors.
The color encodes the number of points used to compute each bar.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


info_input = [
    # Each entry: column name, is it continuous, is it already log10?
    {'column': 'DLC groupe', 'is_continuous': False, 'is_log': False},
    {'column': 'C_content', 'is_continuous': True, 'is_log': False},
    {'column': 'H_content', 'is_continuous': True, 'is_log': False},
    {'column': 'Doped', 'is_continuous': False, 'is_log': False},
    {'column': 'Sp2/Sp3', 'is_continuous': True, 'is_log': False},
    {'column': 'Film elastic modulus (GPa)', 'is_continuous': True, 'is_log': False},
    {'column': 'Film hardness (GPa)', 'is_continuous': True, 'is_log': False},
    {'column': 'Ball elastic modulus (GPa)', 'is_continuous': True, 'is_log': False},
    {'column': 'Ball hardness (GPa)', 'is_continuous': True, 'is_log': False},
    {'column': 'Load (N)', 'is_continuous': True, 'is_log': False},
    {'column': 'Temperature', 'is_continuous': True, 'is_log': False},
    {'column': 'Humidity', 'is_continuous': False, 'is_log': False},
    {'column': 'E*', 'is_continuous': True, 'is_log': False},
    {'column': 'Hertz pressure (Gpa)', 'is_continuous': True, 'is_log': False},
    {'column': 'log10(Sliding distance (m))', 'is_continuous': True, 'is_log': True},
    {'column': 'log10(Wear Volume V)', 'is_continuous': True, 'is_log': True},
    {'column': 'log10(Sliding velocity (m/s))', 'is_continuous': True, 'is_log': True},
    {'column': 'log10(Rq (nm))', 'is_continuous': True, 'is_log': True},
]

# =========================
# Chargement des données
# =========================
df = pd.read_csv(
    "data/cleaned_dataset.csv",
    sep=";",
    decimal=",",
    encoding="utf-8"
)

for info in info_input:

    # Parameters
    col_name = info['column']
    is_log = info.get('is_log', False)
    is_continuous = info.get('is_continuous', False)
    n_bins = 50

    # Prepare zone/bin column
    if is_continuous:
        val_min = df[col_name].min()
        val_max = df[col_name].max()
        bin_edges = np.linspace(val_min, val_max, n_bins + 1)

        df[col_name + '_zone'] = pd.cut(
            df[col_name],
            bins=bin_edges,
            labels=range(n_bins),
            include_lowest=True,
        )
    else:
        df[col_name + '_zone'] = df[col_name]

    # Build aggregated dataframe per zone
    if is_continuous:
        df_cut = pd.DataFrame({col_name + '_zone': range(n_bins)})
    else:
        df_cut = pd.DataFrame({col_name + '_zone': df[col_name].dropna().unique()})

    # Statistics per bin
    df_cut['mean_log10_wear_rate'] = [
        df.loc[df[col_name + '_zone'] == z, 'log10(Wear rate)'].mean()
        for z in df_cut[col_name + '_zone']
    ]
    df_cut['std_log10_wear_rate'] = [
        df.loc[df[col_name + '_zone'] == z, 'log10(Wear rate)'].std()
        for z in df_cut[col_name + '_zone']
    ]
    df_cut['wear_rate_mean_geometric'] = 10 ** df_cut['mean_log10_wear_rate']

    df_cut['mean_friction_coef'] = [
        df.loc[df[col_name + '_zone'] == z, 'Friction coefficient'].mean()
        for z in df_cut[col_name + '_zone']
    ]
    df_cut['std_friction_coef'] = [
        df.loc[df[col_name + '_zone'] == z, 'Friction coefficient'].std()
        for z in df_cut[col_name + '_zone']
    ]

    df_cut['n_points'] = [
        df.loc[df[col_name + '_zone'] == z, col_name].count()
        for z in df_cut[col_name + '_zone']
    ]

    # Remove empty bins for categorical inputs
    if not is_continuous:
        df_cut = df_cut[df_cut['n_points'] > 0]

    # X positions and widths
    if is_continuous:
        df_cut['x_left'] = bin_edges[:-1]
        df_cut['x_right'] = bin_edges[1:]
        if is_log:
            df_cut['x_plot_left'] = 10 ** df_cut['x_left']
            df_cut['x_plot_right'] = 10 ** df_cut['x_right']
        else:
            df_cut['x_plot_left'] = df_cut['x_left']
            df_cut['x_plot_right'] = df_cut['x_right']
        df_cut['width'] = df_cut['x_plot_right'] - df_cut['x_plot_left']
    else:
        df_cut['x_plot_left'] = np.arange(len(df_cut))
        df_cut['width'] = 0.8

    # Colors by number of points
    norm = Normalize(vmin=df_cut['n_points'].min(), vmax=df_cut['n_points'].max())
    cmap = plt.cm.viridis
    colors = cmap(norm(df_cut['n_points'].values))

    # Error bars for wear rate (geometric mean)
    err_low = (
        10 ** df_cut['mean_log10_wear_rate']
        - 10 ** (df_cut['mean_log10_wear_rate'] - df_cut['std_log10_wear_rate'])
    )
    err_high = (
        10 ** (df_cut['mean_log10_wear_rate'] + df_cut['std_log10_wear_rate'])
        - 10 ** df_cut['mean_log10_wear_rate']
    )
    yerr_wear = np.vstack([err_low.fillna(0).values, err_high.fillna(0).values])

    # Error bars for friction coefficient
    mean_fric = df_cut['mean_friction_coef']
    std_fric = df_cut['std_friction_coef']
    err_low_fric = np.minimum(std_fric.fillna(0), mean_fric.fillna(0))
    err_high_fric = std_fric.fillna(0)
    yerr_fric = np.vstack([err_low_fric.values, err_high_fric.values])

    # Figure
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    # Wear rate panel
    ax0.bar(
        df_cut['x_plot_left'],
        df_cut['wear_rate_mean_geometric'],
        width=df_cut['width'],
        align='edge',
        color=colors,
    )
    ax0.errorbar(
        df_cut['x_plot_left'] + df_cut['width'] / 2,
        df_cut['wear_rate_mean_geometric'],
        yerr=yerr_wear,
        fmt='none',
        ecolor='black',
        elinewidth=0.8,
        capsize=2,
    )
    ax0.set_ylabel("Wear rate")
    ax0.set_yscale('log')

    # Friction coefficient panel
    ax1.bar(
        df_cut['x_plot_left'],
        df_cut['mean_friction_coef'],
        width=df_cut['width'],
        align='edge',
        color=colors,
    )
    ax1.errorbar(
        df_cut['x_plot_left'] + df_cut['width'] / 2,
        mean_fric,
        yerr=yerr_fric,
        fmt='none',
        ecolor='black',
        elinewidth=0.8,
        capsize=2,
    )
    ax1.set_ylabel("Friction coefficient")

    # Axes labels
    if is_log:
        ax0.set_xscale('log')
        ax1.set_xscale('log')
        xlabel = col_name[6:-1]
    else:
        xlabel = col_name
    ax0.set_xlabel(xlabel)
    ax1.set_xlabel(xlabel)

    ax0.set_title("Geometric mean of Wear rate")
    ax1.set_title("Mean friction coefficient")

    if not is_continuous:
        xticks = df_cut['x_plot_left'] + df_cut['width'] / 2
        labels = df_cut[col_name + '_zone']
        ax0.set_xticks(xticks)
        ax0.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(labels, rotation=45, ha='right')

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(df_cut['n_points'].values)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Number of points per bin")

    plt.subplots_adjust(wspace=0.4)

    fname = (col_name[6:-1] if is_log else col_name).replace('/', '_').replace(' ', '_')

    plt.savefig(
        f'make_figures/figures/Fric_wear_input/Fric_wear_{fname}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.close()
    print(f'{col_name} done')

print('TASK FINISHED')