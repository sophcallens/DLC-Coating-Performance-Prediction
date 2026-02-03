import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from log_scale import log10_ticks

# =========================
# Paramètres de la figure
# =========================
info_input = [
    {'input1': 'Temperature', 'input1_log': False,
     'input2': 'Humidity', 'input2_log': False},
    {'input1': 'Film elastic modulus (GPa)', 'input1_log': False,
     'input2': 'Film hardness (GPa)', 'input2_log': False},
    {'input1': 'log10(Sliding distance (m))', 'input1_log': True,
     'input2': 'log10(Sliding velocity (m/s))', 'input2_log': True},
    {'input1': 'Friction coefficient', 'input1_log': False,
     'input2': 'Hertz pressure (Gpa)', 'input2_log': False}
]

for info in info_input:

    input1 = info['input1']
    input1_log = info['input1_log']
    input2 = info['input2']
    input2_log = info['input2_log']

    # =========================
    # Chargement et nettoyage
    # =========================
    df = pd.read_csv(
        "data/cleaned_dataset.csv",
        sep=";",
        decimal=",",
        encoding="utf-8"
    )
    df = df.dropna(subset=[input1, input2])

    x = df[input1].values
    y = df[input2].values

    # =========================
    # Hexbin = nombre de points
    # =========================
    plt.figure(figsize=(8,6))

    hb = plt.hexbin(
        x, y,
        gridsize=40,
        cmap='viridis',
        mincnt=1,
        linewidths=0.1,      # contour très fin
        edgecolors='none'    # ou 'k' si tu veux les voir
    )

    ax = plt.gca()

    if input1_log:
        log10_ticks(ax, axis="x")

    if input2_log:
        log10_ticks(ax, axis="y")

    plt.colorbar(hb, label='Nombre de points')
    plt.xlabel(input1)
    plt.ylabel(input2)
    plt.title('Densité = nombre de points')

    # =========================
    # Sauvegarde
    # =========================
    input1_fname = input1.replace("/", "_").replace(" ", "_")
    input2_fname = input2.replace("/", "_").replace(" ", "_")

    plt.savefig(
        f'make_figures/figures/input1_input2_density/{input1_fname}_{input2_fname}_density.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()

    print(f'{input1_fname}_{input2_fname} done')

print('TASK FINISHED')
