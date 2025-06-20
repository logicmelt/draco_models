from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from draco_models.config import InputConfig
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Any

AVAILABLE_MODELS = {
    "KNeighborsRegressor": KNeighborsRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "SVR": SVR(),
    "SGDRegressor": SGDRegressor(),
}


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
        pipeline = Pipeline(
            [("scaler", StandardScaler()), (model_name, model_instance)]
        )
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
