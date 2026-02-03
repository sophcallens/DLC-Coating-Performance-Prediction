"""Ternary composition plots

Create ternary diagrams using C, H, Sp2 fractions and color/shape by a
selected input column. Adds legends for both the input (color) and the
`Doped` status (marker shape).

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import ternary
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D


info_input = [
    {'column': 'DLC groupe', 'is_continuous': False},
    {'column': 'Film hardness (GPa)', 'is_continuous': True},
    {'column': 'Friction coefficient', 'is_continuous': True},
]


for info in info_input:

    col_name = info['column']
    is_continuous = info.get('is_continuous', False)

    # Load data
    df = pd.read_csv(
        "data/cleaned_dataset.csv",
        sep=";",
        decimal=",",
        encoding="utf-8"
    )

    df_tri = df[['C_content', 'H_content', 'Sp2/Sp3', 'Doped', col_name]].dropna().copy()

    # Compute ternary fractions (H, Sp3, Sp2) from C/H and Sp2/Sp3
    h_frac = df_tri['H_content'] / (df_tri['H_content'] + df_tri['C_content'])
    sp2_frac = (df_tri['Sp2/Sp3'] / (1 + df_tri['Sp2/Sp3'])) * (1 - h_frac)
    sp3_frac = (1 - df_tri['Sp2/Sp3'] / (1 + df_tri['Sp2/Sp3'])) * (1 - h_frac)

    df_tri['H'] = h_frac
    df_tri['Sp2'] = sp2_frac
    df_tri['Sp3'] = sp3_frac

    # Ternary figure
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary(linewidth=2)
    tax.gridlines(color="gray", multiple=0.1)
    figure.set_frameon(False)
    plt.axis('off')

    # Corner labels
    tax.top_corner_label("sp³", fontsize=12)
    tax.left_corner_label("sp²", fontsize=12)
    tax.right_corner_label("H", fontsize=12)

    # Color and marker setup
    shapes = {False: 'o', True: '^'}  # not doped: circle, doped: triangle

    if is_continuous:
        norm = Normalize(vmin=df_tri[col_name].min(), vmax=df_tri[col_name].max())
        cmap = plt.cm.viridis
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(df_tri[col_name].values)
        plt.colorbar(sm, ax=tax.get_axes(), shrink=0.7, pad=0.02, label=col_name)
    else:
        values = sorted(df_tri[col_name].unique())
        cmap = plt.get_cmap("viridis", len(values))
        colors_dict = {value: cmap(i) for i, value in enumerate(values)}

    # Scatter points
    for _, row in df_tri.iterrows():
        if pd.isna(row[col_name]) or pd.isna(row['Doped']):
            continue

        if is_continuous:
            c = cmap(norm(row[col_name]))
        else:
            c = colors_dict.get(row[col_name], "gray")

        m = shapes.get(bool(row['Doped']), 'o')

        tax.scatter([(row['H'], row['Sp3'], row['Sp2'])], color=c, marker=m, s=50, alpha=0.6)

    # Legends
    ax = plt.gca()
    if not is_continuous:
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=str(val),
                   markerfacecolor=colors_dict.get(val, 'gray'), markersize=8)
            for val in values
        ]
        ax.legend(handles=legend_elements, title=col_name, loc='upper right')

    doped_handles = [
        Line2D([0], [0], marker=shapes[False], color='k', label='Not doped', linestyle=''),
        Line2D([0], [0], marker=shapes[True], color='k', label='Doped', linestyle='')
    ]
    ax.add_artist(ax.legend(handles=doped_handles, loc='upper left', title='Doped status'))

    # Final touches and save
    tax.ticks(axis='lbr', linewidth=1, multiple=0.1, tick_formats="%.1f", fontsize=5)
    tax.clear_matplotlib_ticks()

    fname = col_name.replace('/', '_').replace(' ', '_')
    plt.savefig(f'make_figures/figures/Ternary_diagram/Ternary_diagram_{fname}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f'{col_name} done')

print('TASK FINISHED')