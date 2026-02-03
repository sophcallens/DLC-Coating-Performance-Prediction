"""Scatter + KDE by DLC family

For each DLC family (column `DLC groupe`) plot input1 vs input2 with
KDE-filled contours to show typical zones. Colors encode families.

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from log_scale import log10_ticks

info_input = [
    {'input1': 'Film hardness (GPa)', 'input1_log': False, 'input2': 'Sp2/Sp3', 'input2_log': False},
    {'input1': 'Film elastic modulus (GPa)', 'input1_log': False, 'input2': 'Sp2/Sp3', 'input2_log': False},
]

for info in info_input:

    input1 = info['input1']
    input1_log = info['input1_log']
    input2 = info['input2']
    input2_log = info['input2_log']

    # Load and filter
    df = pd.read_csv("data/cleaned_dataset.csv", sep=";", decimal=",", encoding="utf-8")
    df = df.dropna(subset=[input1, input2, 'DLC groupe'])

    families = df['DLC groupe'].unique()
    cmap = plt.get_cmap("viridis", len(families))
    colors = {value: cmap(i) for i, value in enumerate(families)}

    plt.figure(figsize=(8, 6))

    for fam in families:
        subset = df[df['DLC groupe'] == fam][[input1, input2]].dropna()

        # Scatter for each family
        plt.scatter(subset[input1], subset[input2], color=colors[fam], label=fam, s=50, alpha=0.7)

        # KDE to show a semi-transparent average zone
        if len(subset) >= 3:
            sns.kdeplot(
                x=subset[input1],
                y=subset[input2],
                fill=True,
                alpha=0.2,
                levels=2,
                color=colors[fam],
                linewidths=0,
            )

    ax = plt.gca()

    if input1_log:
        log10_ticks(ax, axis="x")

    if input2_log:
        log10_ticks(ax, axis="y")

    plt.xlabel(input1)
    plt.ylabel(input2)
    plt.title("Points grouped by DLC family with average zones")
    plt.legend(title="DLC group")

    # Save
    input1_fname = input1.replace("/", "_").replace(" ", "_")
    input2_fname = input2.replace("/", "_").replace(" ", "_")

    out_path = f'make_figures/figures/input1_input2_family/{input1_fname}_{input2_fname}_family.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'{input1_fname}_{input2_fname} done')


print('TASK FINISHED')