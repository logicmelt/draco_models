import pydantic
import pathlib
import yaml
import json
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


class InfluxDBConfig(pydantic.BaseModel):
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


class InputConfig(pydantic.BaseModel):
    """Input configuration for the training pipeline."""

    influxdb: InfluxDBConfig = pydantic.Field(
        default_factory=InfluxDBConfig,
        description="Configuration for connecting to the InfluxDB instance.",
    )
    query: str = pydantic.Field(
        default='from(bucket: "logicmelt") |> range(start: -365d)\
        |> filter(fn: (r) => r["_measurement"] == "particle")',
        description="The Flux query to execute against the InfluxDB instance.",
    )
    time_resolution: float = pydantic.Field(
        default=1.0,
        description="The time resolution in seconds for the data aggregation.",
    )
    density_profile: pathlib.Path = pydantic.Field(
        default=pathlib.Path("additional_files/density_temp_height.json"),
        description="Path to the CSV file containing the density profile data in json format.",
    )
