import pandas as pd
import numpy as np
from src.models import get_all_pipelines, CATEGORICAL_COLUMNS, BOOL_COLUMNS
from src.utils import evaluate_model
import warnings
import time

from sklearn.model_selection import RepeatedKFold
from sklearn.base import clone

warnings.filterwarnings("ignore")  # éviter les warnings sklearn

# ----------------------------
# Charger les données
# ----------------------------
df = pd.read_csv(
    "project/data/selected/Scenario2.csv",
    sep=";",
    decimal=",",
    encoding="utf-8"
)
test1 = ['Friction coefficient'] # seulement pour Cof (perte de 80 lignes) -> 2158 lignes
test2 = ['log10(Wear rate)'] # seulement pour Wear rate (perte de 702 lignes) -> ? lignes
test3 =['Friction coefficient', 'log10(Wear rate)'] # pour les deux (perte de plus de 700 lignes) -> ? lignes

# ----------------------------
# Séparer X / y
# ----------------------------
df_test = df.dropna(subset=test1)
print(len(df_test))
X = df_test.drop(columns=["Friction coefficient", "log10(Wear rate)"])
y = df_test[["Friction coefficient"]]

# ----------------------------
# Détection automatique des colonnes catégorielles / bool
# ----------------------------
categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in X.columns]
bool_columns = [col for col in BOOL_COLUMNS if col in X.columns]

print("Colonnes catégorielles détectées :", categorical_columns)
print("Colonnes booléennes détectées :", bool_columns)

# ----------------------------
# Récupérer tous les pipelines
# ----------------------------
pipelines = get_all_pipelines(categorical_columns, bool_columns)

# ----------------------------
# Paramètres CV
# ----------------------------
N_SPLITS = 5
N_REPEATS = 3
RANDOM_STATE = 42

cv = RepeatedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=RANDOM_STATE
)

# ----------------------------
# Boucler sur chaque pipeline et calculer métriques
# ----------------------------
results = []

for name, pipeline in pipelines.items():
    rmse_scores = []
    r2_scores = []

    print(f"\n🔍 Pipeline : {name}")
    start_time = time.time()
    
    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipe = clone(pipeline)
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        rmse_scores.append(metrics["RMSE"])
        r2_scores.append(metrics["R2"])

        print(
            f"  Fold {fold+1:02d} | "
            f"RMSE = {metrics['RMSE']:.3f} | "
            f"R2 = {metrics['R2']:.3f}"
        )
    end_time = time.time()
    results.append({
        "pipeline": name,
        "RMSE_mean": np.mean(rmse_scores),
        "RMSE_std": np.std(rmse_scores),
        "R2_mean": np.mean(r2_scores),
        "R2_std": np.std(r2_scores),
        "time": end_time - start_time
    })

# ----------------------------
# Sauvegarde des résultats
# ----------------------------
results_df = pd.DataFrame(results).sort_values(by="R2_mean", ascending=False)
results_df.to_csv("project/pred_comparison/results/metrics/Scenario2.csv",  sep=";", decimal=",", index=False, encoding="utf-8")

print("\n✅ Toutes les pipelines ont été testées.")
print("📁 Résultats sauvegardés dans project/results/metrics/pipeline_results.csv")
