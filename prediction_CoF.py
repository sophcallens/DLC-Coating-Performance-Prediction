# 1. Sélection de variables pertinentes
# 2. Analyse physique des données manquantes
# 3. Imputation multiple (MICE / MissForest)
# 4. Modèle robuste (XGBoost / RF)
# 5. Modèles séparés CoF / Wear
# 6. Validation par régime + incertitude

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import traceback
from tqdm import tqdm


############
# Importation des données
############
def data_importation():
    df = pd.read_csv("data/cleaned_dataset.csv",sep=";",decimal=",",encoding="utf-8")
    return df.dropna(subset=['Friction coefficient'])



############
# Sélection de variables pertinentes
############

def variable_selection(df, scenario):
    if scenario == 1:
        cols = ['log10(Sliding velocity (m/s))','Humidity','Ball hardness (GPa)','Load (N)','Temperature','Sp2/Sp3','DLC groupe','Film hardness (GPa)']
    elif scenario == 2:
        cols = ['log10(Sliding velocity (m/s))','Humidity','Ball hardness (GPa)','Load (N)','Temperature','Sp2/Sp3','DLC groupe','Film hardness (GPa)','Doped','H']
    else:
        raise ValueError(f'scenario {scenario} does not exist, choose 1 or 2')
    
    X = df[cols]
    y = df['Friction coefficient']
    return X, y



############
# Encoder pour n'avoir que des int
############

def encode(X: pd.DataFrame, column: str) -> pd.DataFrame:
    X_encoded = pd.DataFrame(index=X.index)  # même index pour aligner
    for valeur in X[column].dropna().unique():
        X_encoded[valeur] = X[column].apply(lambda x: 1 if x == valeur else (0 if pd.notna(x) else np.nan)) # 1 si la valeur correspond, 0 si non, NaN si la valeur originale est NaN
    return pd.concat([X.drop(columns=[column]), X_encoded], axis=1)

############
# Normalisation des données
############

def normalize(X: pd.DataFrame, do: bool) -> pd.DataFrame:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    if do:
        return pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    else:
        return pd.DataFrame(scaler.inverse_transform(X), columns=X.columns)

############
# Analyse physique des données manquantes
############
# On n'importe plus 10 fonctions, mais juste la classe "DataImputer"
from Imputation import DataImputer 

############
# MAIN scenario 1
############

"""0. Importation des données"""
df = data_importation()

"""1. Sélection de variables pertinentes"""
X1, y1 = variable_selection(df, scenario=1)

"""2. Encoder les données non numériques"""
X1_encoded = encode(X1, column='DLC groupe')

"""3. Normalisation des données"""
X1_normalized = normalize(X1_encoded, do=True)

"""4. Analyse physique des données manquantes (Version Classe)"""

imputer_tool = DataImputer(random_state=0)

# comparison
methods_to_test = [imputer_tool.simple_median, imputer_tool.knn, imputer_tool.mice,imputer_tool.random_forest,imputer_tool.extra_trees]

imputer_tool.compare_all(X1_normalized, list_of_methods=methods_to_test)
imputer_tool.get_comparaison_summary()

# results :
#                         imputation  mean_mae  mean_rmse
# 0         imputation_simple_median  0.598696   1.093603
# 1       imputation_semi_simple_KNN  0.291834   0.581655
# 2           imputatio_avancée_MICE  0.474030   0.700945
# 3  imputation_avancée_RandomForest  0.281625   0.551107
# 4    imputation_avancée_ExtraTrees  0.218903   0.449825
# -> on garde ExtraTrees (même si très très lent)

"""5. application de la meilleure méthode trouvée"""
X1_final = imputer_tool.extra_trees(X1_normalized)


#optimisation des hyperparamètres



# bis. Dénormalisation des données
