"""Element-wise comparison plots

Produce side-by-side bar plots for two inputs across a set of elements.
The color encodes the number of points used to compute each bar.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# Fill with the pair(s) of columns to plot. Each entry describes: col1, is_col1_log, col2, is_col2_log
info_input = [
    {'col1': 'log10(Wear rate)', 'col1_log': True, 'col2': 'Friction coefficient', 'col2_log': False},
    {'col1': 'Film hardness (GPa)', 'col1_log': False, 'col2': 'Film elastic modulus (GPa)', 'col2_log': False}

]

for info in info_input:

    # Parameters
    col1 = info['col1']
    col1_log = info.get('col1_log', False)
    col2 = info['col2']
    col2_log = info.get('col2_log', False)
    elements = ['O', 'Ti', 'Si', 'H', 'W', 'Mo', 'N', 'Cr', 'Ta', 'Ag', 'Ar', 'Al', 'Fe', 'Nb', 'Cu', 'F', 'S', 'B', 'Ne']

    # Load data
    df = pd.read_csv(
        "data/cleaned_dataset.csv",
        sep=";",
        decimal=",",
        encoding="utf-8"
    )

    # Aggregate statistics per element
    df_element = pd.DataFrame({'elements': elements})

    def get_data_stats(col_name):
        mean_list, std_list, npoints_list = [], [], []
        for el in elements:
            data = df.loc[df[el] == True, col_name]
            mean_list.append(data.mean())
            std_list.append(data.std())
            npoints_list.append(data.count())
        return mean_list, std_list, npoints_list

    mean1, std1, npoints1 = get_data_stats(col1)
    df_element['mean1'] = mean1
    df_element['std1'] = std1
    df_element['npoints1'] = npoints1
    if col1_log:
        df_element['geom_mean1'] = 10 ** df_element['mean1']

    mean2, std2, npoints2 = get_data_stats(col2)
    df_element['mean2'] = mean2
    df_element['std2'] = std2
    df_element['npoints2'] = npoints2
    if col2_log:
        df_element['geom_mean2'] = 10 ** df_element['mean2']

    # Filter elements without data
    df_el_f1 = df_element[df_element['npoints1'] > 0].copy()
    df_el_f2 = df_element[df_element['npoints2'] > 0].copy()

    df_el_s1 = df_el_f1.sort_values(by='mean1', ascending=True)
    df_el_s2 = df_el_f2.sort_values(by='mean2', ascending=True)

    # Colors by number of points
    norm = Normalize(vmin=df_el_s1['npoints1'].min(), vmax=df_el_s1['npoints1'].max())
    cmap = plt.cm.viridis
    colors1 = cmap(norm(df_el_s1['npoints1']))
    colors2 = cmap(norm(df_el_s2['npoints2']))

    # Error bars
    if col1_log:
        err_low1 = 10 ** df_el_s1['mean1'] - 10 ** (df_el_s1['mean1'] - df_el_s1['std1'])
        err_high1 = 10 ** (df_el_s1['mean1'] + df_el_s1['std1']) - 10 ** df_el_s1['mean1']
        yerr1 = [err_low1.fillna(0).values, err_high1.fillna(0).values]
    else:
        yerr1 = df_el_s1['std1'].fillna(0).values

    if col2_log:
        err_low2 = 10 ** df_el_s2['mean2'] - 10 ** (df_el_s2['mean2'] - df_el_s2['std2'])
        err_high2 = 10 ** (df_el_s2['mean2'] + df_el_s2['std2']) - 10 ** df_el_s2['mean2']
        yerr2 = [err_low2.fillna(0).values, err_high2.fillna(0).values]
    else:
        yerr2 = df_el_s2['std2'].fillna(0).values

    # Figure
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    cax = fig.add_subplot(gs[2])

    # Input1
    ax0.bar(
        df_el_s1['elements'],
        df_el_s1['geom_mean1'] if col1_log else df_el_s1['mean1'],
        yerr=yerr1,
        color=colors1,
        capsize=5,
    )
    ax0.set_title(f"{('Geometric mean' if col1_log else 'Mean')} {col1[6:-1] if col1_log else col1} by element")
    ax0.set_xlabel("Elements")
    ax0.set_ylabel(f"{('Geometric mean' if col1_log else 'Mean')} {col1[6:-1] if col1_log else col1}")
    if col1_log:
        ax0.set_yscale('log')
    ax0.tick_params(axis='x', rotation=45)

    # Input2
    ax1.bar(
        df_el_s2['elements'],
        df_el_s2['geom_mean2'] if col2_log else df_el_s2['mean2'],
        yerr=yerr2,
        color=colors2,
        capsize=5,
    )
    ax1.set_title(f"{('Geometric mean' if col2_log else 'Mean')} {col2[6:-1] if col2_log else col2} by element")
    ax1.set_xlabel("Elements")
    ax1.set_ylabel(f"{('Geometric mean' if col2_log else 'Mean')} {col2[6:-1] if col2_log else col2}")
    if col2_log:
        ax1.set_yscale('log')
    ax1.tick_params(axis='x', rotation=45)

    # Colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(df_el_s1['npoints1'].values)
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Number of points per element")

    plt.subplots_adjust(wspace=0.4)

    out_dir = 'make_figures/figures/input1_input2_compo'
    name1 = (col1[6:-1] if col1_log else col1).replace('/', '_').replace(' ', '_')
    name2 = (col2[6:-1] if col2_log else col2).replace('/', '_').replace(' ', '_')
    plt.savefig(f"{out_dir}/{name1}_{name2}_compo.png")
    plt.close()
    print(f'{col1}_{col2} done')

print('TASK FINISHED')