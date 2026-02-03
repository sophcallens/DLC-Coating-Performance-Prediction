"""Presence per feature

creates a graph of number of presence of data for each feature

"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Chargement des données
df = pd.read_csv(
    "data/cleaned_dataset.csv",
    sep=";",
    decimal=",",
    encoding="utf-8"
)

# Liste des paramètres
df_param = pd.DataFrame({'Parameters': df.columns})

# Nombre de points non NaN par paramètre
df_param['N points'] = [
    df[param].notna().sum()
    for param in df_param['Parameters']
]

# Tri par nombre de points
df_param_sorted = df_param.sort_values(
    by='N points',
    ascending=True
)

# Couleurs selon N points
norm = Normalize(
    vmin=df_param_sorted['N points'].min(),
    vmax=df_param_sorted['N points'].max()
)
cmap = plt.cm.jet
colors = cmap(norm(df_param_sorted['N points']))

# Figure
fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.bar(
    df_param_sorted['Parameters'],
    df_param_sorted['N points'],
    color=colors
)

ax.set_title("Nombre de données disponibles par paramètre")
ax.set_xlabel("Paramètres")
ax.set_ylabel("Nombre de points")
ax.tick_params(axis='x', rotation=90)

# Colorbar liée à l'axe
sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("Nombre de points")

plt.tight_layout()


# Save
plt.savefig(f'make_figures/figures/presence_per_feartures.png', dpi=300, bbox_inches='tight')
plt.close()
