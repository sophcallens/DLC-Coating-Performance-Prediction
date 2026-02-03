import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

# Charger le csv
df = pd.read_csv(
    "project/pred_comparison/results/metrics/Scenario3.csv",
    sep=";",
    decimal=",",
    encoding="utf-8"
)

# Parser le nom du pipeline
def parse_pipeline(name):
    parts = name.split("__")
    return {
        "imputer": parts[0],
        "model": parts[1],
        "scaler": parts[2].replace("scaler", ""),  # True / False
        "encoder": parts[3].replace("encoder", "")
    }

parsed = df["pipeline"].apply(parse_pipeline).apply(pd.Series)
df = pd.concat([df, parsed], axis=1)

# ----------------------------------------------------
# Préparation des axes : créer colonne pour chaque imputer + scaler
# ----------------------------------------------------
imputers = sorted(df["imputer"].unique())
scalers = ["True", "False"]
columns = [f"{imp}\nscaler{sc}" for imp in imputers for sc in scalers]

models = sorted(df["model"].unique())
n_rows = len(models)
n_cols = len(columns)

# Matrices pour le tableau
cell_text = [["" for _ in range(n_cols)] for _ in range(n_rows)]
time_matrix = np.full((n_rows, n_cols), np.nan)

# ----------------------------------------------------
# Remplissage
# ----------------------------------------------------
for _, row in df.iterrows():
    i = models.index(row["model"])
    j = columns.index(f"{row['imputer']}\nscaler{row['scaler']}")

    ligne1 = f"RMSE:{row['RMSE_mean']:.3f}"
    ligne2 = f"R2:{row['R2_mean']:.3f}"

    if row['R2_mean'] > 0.5 :
        ligne1 = r"$\bf{" + ligne1 + "}$"
        ligne2 = r"$\bf{" + ligne2 + "}$"

    text = ligne1 + "\n" + ligne2

    cell_text[i][j] = text
    time_matrix[i, j] = row["time"]

# ----------------------------------------------------
# Normalisation des couleurs (time)
# ----------------------------------------------------
norm = mcolors.Normalize(
    vmin=np.nanmin(time_matrix),
    vmax=np.nanmax(time_matrix)
)
cmap = plt.cm.jet
cell_colors = cmap(norm(time_matrix))
cell_colors[..., -1] = 0.5 

# ----------------------------------------------------
# Plot
# ----------------------------------------------------
fig, ax = plt.subplots(figsize=(1.5 * n_cols, 0.8 * n_rows))
ax.axis("off")

# Définir bbox pour déplacer le tableau plus bas et à droite
table = ax.table(
    cellText=cell_text,
    cellColours=cell_colors,
    rowLabels=models,
    colLabels=columns,
    loc="center",
    cellLoc="center",
    bbox=[0.15, 0.15, 0.7, 0.7]  # [x0, y0, largeur, hauteur]
)

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 1.6)

# ----------------------------------------------------
# Ajouter titres colonnes et lignes
# ----------------------------------------------------
# Ensuite, pour les titres, on les met par rapport au tableau lui-même
# Colonnes : au-dessus du tableau
ax.text(0.5, 0.88, "Imputer + Scaler", ha='center', va='bottom',
        fontsize=12, fontweight='bold', transform=ax.transAxes)

# Lignes : à gauche du tableau
ax.text(0.05, 0.5, "Model", ha='center', va='center',
        fontsize=12, fontweight='bold', rotation=90, transform=ax.transAxes)


# ----------------------------------------------------
# Colorbar
# ----------------------------------------------------
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("time")

plt.tight_layout()
plt.show()
