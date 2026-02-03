import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

import optuna
import warnings
warnings.filterwarnings("ignore")


# ----------------------------
# Charger les données
# ----------------------------
def load_data(scenario, col_to_predict, reduced: bool):
    df = pd.read_csv(
        f"project/data/selected/Scenario{scenario}.csv",
        sep=";",
        decimal=",",
        encoding="utf-8"
    )

    df = df.dropna(subset=[col_to_predict])

    if reduced and col_to_predict == "Friction coefficient":
        df = df[df["Friction coefficient"] < 0.4]
    if reduced and col_to_predict == "log10(Wear rate)":
        df = df[df["log10(Wear rate)"] > -9.5]
    return df


# ----------------------------
# Séparer X / y
# ----------------------------
def separate_data(df, col_to_predict):
    X = df.drop(columns=["Friction coefficient", "log10(Wear rate)"])
    y = df[col_to_predict]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )


# ----------------------------
# Definir de la fonction objectif 
# ----------------------------
def make_objective(X_train, y_train):
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 30),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical(
                "max_features", ["sqrt", "log2", None]
            ),
            "random_state": 42,
            "n_jobs": -1
        }

        model = RandomForestRegressor(**params)
        cv = KFold(n_splits=10, shuffle=True, random_state=42)

        return cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="r2"
        ).mean()

    return objective


# ----------------------------
# Créer et run l'optimisation
# ----------------------------
def optimisation(X_train, y_train, n_trials=100):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        make_objective(X_train, y_train),
        n_trials=n_trials,
        n_jobs=-1,
        show_progress_bar=True
    )

    print("Best parameters:", study.best_params)
    return study, study.best_params


# ----------------------------
# Plotting optimisation history
# ----------------------------
def plot_optim_history(study, out_path):
    values = [t.value for t in study.trials if t.value is not None]
    best_values = np.maximum.accumulate(values)

    plt.figure(figsize=(7, 4))
    plt.plot(values, marker="o", alpha=0.6, label="R² par trial")
    plt.plot(best_values, linestyle="--", label="Meilleur R²")
    plt.xlabel("Trial")
    plt.ylabel("R²")
    plt.title("Optimisation Optuna – Historique du R²")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ----------------------------
# Plotting result
# ----------------------------

def plot_prediction_results(out_dir,X_train, y_train,X_test, y_test,best_params):
    model = RandomForestRegressor(
        **best_params,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # --- Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # --- Prediction plot
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--"
    )

    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title("Random Forest – Prédiction")

    plt.text(
        0.05, 0.95,
        f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round"),
        fontsize=11
    )

    plt.tight_layout()
    plt.savefig(f"{out_dir}_prediction.png", dpi=300)
    plt.close()

    # --- Permutation importance
    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="r2"
    )

    importances = pd.Series(
        result.importances_mean,
        index=X_test.columns
    ).sort_values()

    plt.figure(figsize=(7, 5))
    importances.tail(15).plot.barh()
    plt.axvline(0, color="black")
    plt.xlabel("ΔR² après permutation")
    plt.title("Permutation importance – Random Forest")
    plt.tight_layout()
    plt.savefig(f"{out_dir}_importance.png", dpi=300)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------

os.makedirs("project/pred_rd_forest/results", exist_ok=True)

for scenario in [1, 2, 3]:
    for col_to_predict in ["log10(Wear rate)"]:
        for reduced in [True]:

            tag = "reduced" if reduced else "full"
            target = "coF" if col_to_predict == "Friction coefficient" else "wear"

            out_dir = f"project/pred_rd_forest/results/Scenario{scenario}_{target}_{tag}"

            df = load_data(scenario, col_to_predict, reduced)
            X_train, X_test, y_train, y_test = separate_data(df, col_to_predict)

            study, best_params = optimisation(X_train, y_train)

            plot_optim_history(
                study,
                f"{out_dir}_optim.png"
            )

            plot_prediction_results(
                out_dir,
                X_train, y_train,
                X_test, y_test,
                best_params
            )


'''
for scenario in [1, 2, 3]:
    for col_to_predict in ["Friction coefficient", "log10(Wear rate)"]:
        for reduced in [True, False]:

            tag = "reduced" if reduced else "full"
            target = "coF" if col_to_predict == "Friction coefficient" else "wear"

            out_dir = f"project/pred_rd_forest/results/Scenario{scenario}_{target}_{tag}"

            df = load_data(scenario, col_to_predict, reduced)
            X_train, X_test, y_train, y_test = separate_data(df, col_to_predict)

            study, best_params = optimisation(X_train, y_train)

            plot_optim_history(
                study,
                f"{out_dir}_optim.png"
            )

            plot_prediction_results(
                out_dir,
                X_train, y_train,
                X_test, y_test,
                best_params
            )
'''