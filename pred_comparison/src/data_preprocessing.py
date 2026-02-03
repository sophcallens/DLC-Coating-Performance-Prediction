from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler

def get_imputer(strategy="mean"):
    if strategy == "mean":
        return SimpleImputer(strategy="mean")
    elif strategy == "median":
        return SimpleImputer(strategy="median")
    elif strategy == "knn":
        return KNNImputer()
    elif strategy == "iterative":
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.experimental import enable_iterative_imputer
        return IterativeImputer(estimator=ExtraTreesRegressor(), max_iter=10)
    elif strategy == "passthrough":
        return "passthrough"
    else:
        raise ValueError(f"Imputer inconnu: {strategy}")

def get_scaler():
    return StandardScaler()
