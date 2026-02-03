import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance

import xgboost as xgb
import optuna


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
# Fonction objectif Optuna
# ----------------------------
def make_objective(X_train, y_train):

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),

            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist"
        }

        model = xgb.XGBRegressor(**params)

        scores = cross_val_score(model,X_train,y_train,cv=cv,scoring="neg_root_mean_squared_error")

        return scores.mean()  # RMSE

    return objective


# ----------------------------
# Optimisation Optuna
# ----------------------------
def optimisation(X_train, y_train, n_trials=100):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        make_objective(X_train, y_train),
        n_trials=n_trials,
        n_jobs=-1,
        show_progress_bar=True
    )

    print("Best RMSE:", study.best_value)
    print("Best parameters:", study.best_params)

    return study, study.best_params


# ----------------------------
# Plot optimisation history
# ----------------------------
def plot_optim_history(study, out_path):
    values = [t.value for t in study.trials if t.value is not None]
    best_values = np.minimum.accumulate(values)

    plt.figure(figsize=(7, 4))
    plt.plot(values, marker="o", alpha=0.6, label="RMSE par trial")
    plt.plot(best_values, linestyle="--", label="Meilleur RMSE")
    plt.xlabel("Trial")
    plt.ylabel("RMSE")
    plt.title("Optimisation Optuna – Historique RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


# ----------------------------
# Plot résultats finaux
# ----------------------------
def plot_prediction_results(out_dir, X_train, y_train, X_test, y_test, best_params):

    model = xgb.XGBRegressor(
        **best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
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
    plt.title("XGBoost – Prédiction")

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
    plt.title("Permutation importance – XGBoost")
    plt.tight_layout()
    plt.savefig(f"{out_dir}_importance.png", dpi=300)
    plt.close()


# ----------------------------
# MAIN
# ----------------------------
os.makedirs("project/pred_xgboost/results", exist_ok=True)

for scenario in [1, 2, 3]:
    for col_to_predict in ["Friction coefficient", "log10(Wear rate)"]:
        for reduced in [True, False]:

            tag = "reduced" if reduced else "full"
            target = "coF" if col_to_predict == "Friction coefficient" else "wear"

            out_dir = f"project/pred_xgboost/results/Scenario{scenario}_{target}_{tag}"

            df = load_data(scenario, col_to_predict, reduced)
            X_train, X_test, y_train, y_test = separate_data(df, col_to_predict)

            study, best_params = optimisation(X_train, y_train)

            plot_optim_history(study, f"{out_dir}_optim.png")

            plot_prediction_results(
                out_dir,
                X_train, y_train,
                X_test, y_test,
                best_params
            )
