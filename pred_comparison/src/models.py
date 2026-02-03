from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor

# ----------------------------
# Colonnes connues
# ----------------------------
CATEGORICAL_COLUMNS = ["DLC groupe"]
BOOL_COLUMNS = ["Doped", "H"]

# ----------------------------
# Imputers
# ----------------------------
def get_imputers():
    return {
        "mean": SimpleImputer(strategy="mean"),
        "median": SimpleImputer(strategy="median"),
        "knn": KNNImputer(n_neighbors=5),
        #"iterative_extratrees": IterativeImputer(estimator=ExtraTreesRegressor(n_estimators=50, random_state=42),max_iter=10,random_state=42),
        "passthrough": "passthrough"
    }

# ----------------------------
# Modèles
# ----------------------------
def get_models():
    return {
        "random_forest": MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
        "extra_trees": MultiOutputRegressor(ExtraTreesRegressor(n_estimators=100, random_state=42)),
        "xgboost": MultiOutputRegressor(XGBRegressor(
            n_estimators=100,
            random_state=42,
            objective="reg:squarederror"
        )),
        "elastic_net": MultiOutputRegressor(ElasticNet(random_state=42)),
        "mlp": MultiOutputRegressor(MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=500,
            random_state=42
        ))
    }

# ----------------------------
# Encodeur selon modèle
# ----------------------------
def get_encoder(model_name):
    if model_name in ["random_forest", "extra_trees", "xgboost"]:
        return "ordinal"
    elif model_name in ["elastic_net", "mlp"]:
        return "onehot"
    else:
        return "passthrough"

# ----------------------------
# Compatibilités
# ----------------------------
def _is_compatible(imputer_name, model_name, scaler):
    if imputer_name == "passthrough" and scaler:
        return False
    if imputer_name == "passthrough" and model_name in ["elastic_net", "mlp"]:
        return False
    return True

# ----------------------------
# Pipeline
# ----------------------------
def create_pipeline(
    imputer,
    model,
    model_name,
    scaler,
    categorical_columns,
    bool_columns,
):
    transformers = []

    # Catégoriel
    encoder_type = get_encoder(model_name)

    if categorical_columns:
        if encoder_type == "ordinal":
            transformers.append(
                ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_columns)
            )
        elif encoder_type == "onehot":
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns)
            )

    # Booléens (laisser passer tels quels)
    if bool_columns:
        transformers.append(("bool", "passthrough", bool_columns))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="passthrough"
    )

    steps = [
        ("encoder", preprocessor),
        ("imputer", imputer),
    ]

    if scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append(("model", model))

    return Pipeline(steps)

# ----------------------------
# Toutes les pipelines
# ----------------------------
def get_all_pipelines(categorical_columns,bool_columns):
    pipelines = {}
    imputers = get_imputers()
    models = get_models()

    for imputer_name, imputer in imputers.items():
        for model_name, model in models.items():
            for scaler in [False, True]:
                if _is_compatible(imputer_name, model_name, scaler):
                    encoder_name = get_encoder(model_name)
                    name = f"{imputer_name}__{model_name}__scaler{scaler}__encoder{encoder_name}"

                    pipelines[name] = create_pipeline(
                        imputer=imputer,
                        model=model,
                        model_name=model_name,
                        scaler=scaler,
                        categorical_columns=categorical_columns,
                        bool_columns=bool_columns,
                    )
    return pipelines
