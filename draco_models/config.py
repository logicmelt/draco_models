import pydantic
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
    CliSettingsSource,
)
import pathlib
import yaml
import json
import numpy as np
import re
from typing import Any


def load_config(config_file: str | pathlib.Path) -> dict[str, Any]:
    """Loads a configuration file in YAML or JSON format.

    Args:
        config_file (str | pathlib.Path): The path to the configuration file.

    Returns:
        dict[str, Any]: The configuration file read as a dictionary.
    """
    file_extension = pathlib.Path(config_file).suffix
    if file_extension == ".json":
        with open(config_file, "r") as f:
            config = json.load(f)
    elif file_extension == ".yaml":
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    else:
        raise ValueError("The configuration file must be in JSON or YAML format.")
    return config


def load_config_with_default(config_file: str | pathlib.Path) -> dict[str, Any]:
    """Loads a configuration file in YAML or JSON format and fills missing keys with defaults.

    Args:
        config_file (str | pathlib.Path): The path to the configuration file.

    Returns:
        dict[str, Any]: The configuration file read as a dictionary, with defaults applied.
    """
    config = load_config(config_file)
    default_config = load_config("draco_models/default_config/default.json")
    # Update default_config with the loaded config
    default_config = deep_update(default_config, config)
    return default_config


def deep_update(original: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a dictionary with another dictionary.
    Args:
        original (dict[str, Any]): The original dictionary to be updated.
        update (dict[str, Any]): The dictionary with updates.
    Returns:
        dict[str, Any]: The updated dictionary.
    """
    for key, value in update.items():
        if isinstance(value, dict):
            original[key] = deep_update(original.get(key, {}), value)
        else:
            original[key] = value
    return original


class InfluxDBConfig(BaseSettings):
    """Configuration for connecting to an InfluxDB instance."""

    url: str = pydantic.Field(
        default="http://localhost:8086", description="The URL of the InfluxDB instance."
    )
    token: str = pydantic.Field(
        default="logicmelt",
        description="The authentication token for the InfluxDB instance.",
    )
    org: str = pydantic.Field(
        default="logicmelt", description="The organization name in InfluxDB."
    )
    columns_in: list[str] = pydantic.Field(
        default=[""],
        description="List of columns to include in the parsed output. Leave empty to include all columns.",
    )
    columns_out: list[str] = pydantic.Field(
        default=[""], description="List of columns to exclude from the parsed output."
    )


class TrainConfig(BaseSettings):
    """Configuration for the training pipeline."""

    scoring: tuple[str, ...] = pydantic.Field(
        default=("r2", "neg_root_mean_squared_error"),
        description="Tuple of scoring metrics to be used in the training pipeline from sklearn.",
    )
    n_iter: int = pydantic.Field(
        default=100,
        description="Number of iterations for the randomized search in the training pipeline.",
    )
    refit: str = pydantic.Field(
        default="neg_root_mean_squared_error",
        description="The metric to refit the model on after the randomized search (Sklearn).",
    )
    cv: int | None = pydantic.Field(
        default=5,
        description="Number of cross-validation folds to use in the training pipeline.",
    )
    train_split: float = pydantic.Field(
        default=0.8,
        description="Fraction of the data to use for training. The rest will be used for testing.",
    )
    n_jobs: int = pydantic.Field(
        default=1,
        description="Number of jobs to run in parallel for the training pipeline. -1 means using all processors.",
    )
    shuffle: bool = pydantic.Field(
        default=False,
        description="Whether to shuffle the data before splitting into training and testing sets.",
    )


class AggregatorConfig(BaseSettings):
    """Configuration for the data aggregator."""

    type: str = pydantic.Field(
        default="raw",
        description="The type of aggregator to use. 'raw' for raw data aggregation, 'aggregated' for data already aggregated.",
    )
    time_resolution: float = pydantic.Field(
        default=1.0,
        description="The time resolution in seconds for the data aggregation.",
    )


class InputConfig(BaseSettings):
    """Input configuration for the training pipeline."""

    # Allow the configuration to be parsed from the command line
    model_config = SettingsConfigDict(cli_parse_args=True, env_nested_delimiter="__")

    influxdb: InfluxDBConfig = pydantic.Field(
        default_factory=InfluxDBConfig,
        description="Configuration for connecting to the InfluxDB instance.",
    )
    query: str = pydantic.Field(
        default='from(bucket: "logicmelt") |> range(start: -10d)\
        |> filter(fn: (r) => r["_measurement"] == "particle")',
        description="The Flux query to execute against the InfluxDB instance.",
    )
    aggregator: AggregatorConfig = pydantic.Field(
        default_factory=AggregatorConfig,
        description="Configuration for the data aggregator.",
    )
    density_profile: pathlib.Path = pydantic.Field(
        default=pathlib.Path("additional_files/density_temp_height.json"),
        description="Path to the CSV file containing the density profile data in json format.",
    )
    model: dict[str, dict[str, Any]] = pydantic.Field(
        default_factory=dict,
        description="A dictionary of models to be used in the training pipeline."
        "Each key is the model name and the value is a dictionary of parameters for that model.",
    )
    train_params: TrainConfig = pydantic.Field(
        default_factory=TrainConfig,
        description="Configuration for the training pipeline.",
    )
    save_dir: pathlib.Path = pydantic.Field(
        description="Directory to save the output models and results.",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            env_settings,
            CliSettingsSource(settings_cls, cli_parse_args=True),
            init_settings,
        )

    @pydantic.field_validator("model", mode="before")
    @classmethod
    def parse_model(cls, value: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Parse the model configuration to ensure it is in the correct format."""
        # Change Range(init, end, step) strings to np.arange(init, end, step)
        for key, params in value.items():
            for param_key, param_value in params.items():
                if not isinstance(param_value, str):
                    continue
                if param_value.startswith("Range("):
                    # Use regex to extract the Range parameters
                    match: list[str] = re.findall(r"[-+]?(?:\d*\.*\d+)", param_value)
                    # Get them as floats
                    match_nums = [
                        float(x) if x.find(".") > -1 else int(x) for x in match
                    ]
                    # Create the numpy range
                    parsed_dat = np.arange(*match_nums)
                    # Update the parameter value
                    params[param_key] = parsed_dat
        return value

    @pydantic.field_serializer("model")
    def serialize_model(
        self, value: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Serialize the model configuration to ensure that there are no numpy arrays."""
        # Convert numpy arrays to lists for JSON serialization
        for key, params in value.items():
            for param_key, param_value in params.items():
                if isinstance(param_value, np.ndarray):
                    params[param_key] = param_value.tolist()
        return value
