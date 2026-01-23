import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import inspect
import random

# Imports pour le Bayésien
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args

class DataImputer:
    def __init__(self, random_state=0):
        self.random_state = random_state
        self.comparison_history = {}
        self.best_hyperparam_history = {}
        
        # Le catalogue doit contenir des objets 'Space' pour skopt
        self.SPACE_CATALOG = {
            'n_neighbors': Integer(3, 30, name='n_neighbors'),
            'weights': Categorical(['uniform', 'distance'], name='weights'),
            'max_iter': Integer(10, 50, name='max_iter'),
            'n_estimators': Integer(50, 500, name='n_estimators'),
            'min_samples_leaf': Integer(1, 10, name='min_samples_leaf')
        }

    # --- MÉTHODES D'IMPUTATION ---
    def _to_df(self, array, columns):
        return pd.DataFrame(array, columns=columns)

    def simple_median(self, X):
        imputer = SimpleImputer(strategy="median")
        return self._to_df(imputer.fit_transform(X), X.columns)

    def knn(self, X, n_neighbors=5, weights="distance"):
        imputer = KNNImputer(n_neighbors=int(n_neighbors), weights=weights)
        return self._to_df(imputer.fit_transform(X), X.columns)

    def mice(self, X, max_iter=20):
        imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=int(max_iter), random_state=self.random_state)
        return self._to_df(imputer.fit_transform(X), X.columns)

    def extra_trees(self, X, n_estimators=300, min_samples_leaf=2, max_iter=15):
        imputer = IterativeImputer(
            estimator=ExtraTreesRegressor(n_estimators=int(n_estimators), min_samples_leaf=int(min_samples_leaf), 
                                          random_state=self.random_state, n_jobs=-1),
            max_iter=int(max_iter), random_state=self.random_state
        )
        return self._to_df(imputer.fit_transform(X), X.columns)

    # --- LOGIQUE D'ÉVALUATION ---
    def _evaluate_mae(self, method, X, params):
        """ Évalue une combinaison de paramètres sur un échantillon de colonnes """
        maes = []
        # On limite à 3 colonnes aléatoires pour accélérer l'optimisation
        test_cols = random.sample(list(X.columns), min(3, len(X.columns)))
        
        for column in test_cols:
            X_test = X.copy().dropna(subset=[column])
            if len(X_test) < 10: continue
            
            mask = np.random.rand(len(X_test)) < 0.2
            if mask.sum() == 0: continue
            
            true_values = X_test.loc[mask, column].copy()
            X_test.loc[mask, column] = np.nan
            
            try:
                X_pred_df = method(X_test, **params)
                X_pred = X_pred_df[column].loc[mask]
                maes.append(mean_absolute_error(true_values, X_pred))
            except:
                return 999.0 # Score de pénalité en cas d'erreur de convergence
                
        return np.mean(maes) if maes else 999.0

    # --- OPTIMISATIONS ---
    def get_method_params(self, imputation_method):
        signature = inspect.signature(imputation_method)
        return [p for p in signature.parameters if p not in ['self', 'X']]

    def hyperparam_bayesian(self, imputation_method, X, n_calls=15):
        """ Optimisation Bayésienne intelligente """
        target_params = self.get_method_params(imputation_method)
        search_space = [self.SPACE_CATALOG[p] for p in target_params if p in self.SPACE_CATALOG]
        
        if not search_space:
            print(f"Rien à optimiser pour {imputation_method.__name__}")
            return {}

        @use_named_args(search_space)
        def objective(**params):
            return self._evaluate_mae(imputation_method, X, params)

        res = gp_minimize(objective, search_space, n_calls=n_calls, random_state=self.random_state)
        
        best_params = dict(zip([s.name for s in search_space], res.x))
        self.best_hyperparam_history[imputation_method.__name__] = {"params": best_params, "mae": res.fun}
        return best_params

    def compare_all(self, X_init, methods=None):
        """ Compare plusieurs méthodes et stocke le résultat """
        if methods is None:
            methods = [self.simple_median, self.knn, self.mice]
        
        results = []
        for method in tqdm(methods, desc="Comparaison des méthodes"):
            mae = self._evaluate_mae(method, X_init, {})
            results.append({"imputation": method.__name__, "mean_mae": mae})
            
        df_results = pd.DataFrame(results)
        self.comparison_history['comparisons'] = df_results
        self.comparison_history['best_method'] = df_results.loc[df_results['mean_mae'].idxmin(), 'imputation']
        self.comparison_history['last_run'] = pd.Timestamp.now()
        return df_results