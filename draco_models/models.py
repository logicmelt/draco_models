from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from draco_models.config import InputConfig
from typing import Any

AVAILABLE_MODELS = {
    "KNeighborsRegressor": KNeighborsRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "SVR": SVR(),
    "SGDRegressor": SGDRegressor(),
}


class Scaler(BaseEstimator, TransformerMixin):
    """Custom scaler implementation to integrate with pipeline"""

    _options = {"standard": StandardScaler(), "minmax": MinMaxScaler(), "none": None}

    def __init__(self, option: str = "none"):
        self.option = option

    def fit(self, x, y):
        self.scaler = self.create_scaler()
        self.is_fitted_ = True
        if self.scaler is not None:
            return self.scaler.fit(x, y)
        else:
            return self

    def transform(self, X):
        if self.scaler is not None:
            return self.scaler.transform(X)
        else:
            return X

    def create_scaler(self):
        return self._options[self.option]


def pipeline_factory(config: InputConfig) -> dict[str, Any]:
    """Factory function to create machine learning pipelines based on the provided configuration.

    Args:
        config (InputConfig): The input configuration containing model type and parameters.

    Returns:
        dict[str, Any]: A dictionary containing the model names as keys and their corresponding RandomizedSearchCV objects as values.
    """
    # Iterate over the models in the config and create pipelines for each
    model = {}
    for model_name, model_params in config.model.items():
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model_name} is not available. Choose from {list(AVAILABLE_MODELS.keys())}."
            )

        # Get the model instance
        model_instance = AVAILABLE_MODELS[model_name]

        # Create a pipeline with the model
        pipeline = Pipeline([("scaler", Scaler()), (model_name, model_instance)])
        model_params.update({"scaler__option": ["minmax", "standard", "none"]})
        search = RandomizedSearchCV(
            pipeline,
            model_params,
            n_jobs=config.train_params.n_jobs,
            scoring=config.train_params.scoring,
            refit=config.train_params.refit,
            n_iter=config.train_params.n_iter,
            cv=config.train_params.cv,
        )
        # Store the search object in the model dictionary
        model[model_name] = search
    return model
