from draco_models.config import (
    InputConfig,
    SaveHttpConfig,
    SaveLocalConfig,
    load_config,
)
from draco_models.influx import InfluxDB
from draco_models.aggregator import Aggregator, IdentityAggregator
from draco_models.models import pipeline_factory
from draco_models.utils import create_logger, setup_run_dir
from typing import Any
from skl2onnx import to_onnx
import pathlib
import numpy as np
import json
import hashlib
import uuid
import requests


class Job:
    """Job class to handle the data processing and model training pipeline."""

    def __init__(self, config: InputConfig):
        """Initialize the Job class.
        Args:
            config (InputConfig): The input configuration containing the parameters for the job.
        """
        # Set variables and store the configuration
        self.train_idx: np.ndarray
        self.test_idx: np.ndarray
        self.config = config
        # Set up logging
        self.logger = create_logger("draco_trainer", self.config.logging)
        # Set up the job.
        # Get the data from InfluxDB using the provided query and columns.
        self.influxdb = InfluxDB(config.influxdb)
        self.logger.info("Fetching data from InfluxDB...")
        self.data = self.influxdb.custom_query(
            config.query
        )
        self.logger.info("Data fetched successfully.")
        # Get the density profile file path
        self.logger.info("Parsing density profile...")
        self.atmo_prof = self.parse_density_prof(config.density_profile)
        self.logger.info("Density profile parsed successfully.")
        # Define the aggregator to process the data
        self.logger.info(f"Setting up aggregator: {config.aggregator.type}")
        self.aggregator = (
            Aggregator(config.aggregator.time_resolution)
            if config.aggregator.type == "raw"
            else IdentityAggregator()
        )
        # Now, we have to aggregate the data in time_resolution steps
        self.agg_data, self.target_idx = self.aggregator.aggregate_time_resolution(
            self.data
        )

        # And now we can split the data into training and testing sets
        self.logger.info("Splitting data into training and testing sets.")
        self.train_data, self.test_data = self.train_test_split(
            self.config.train_params.train_split
        )
        self.logger.info(f"Training set size: {self.train_data.shape}")
        # Get the targets ready for training
        self.target_arr = self.get_targets(self.train_idx)
        self.test_arr = self.get_targets(self.test_idx)
        # Prepare the pipeline for training. This is a dictionary of pipelines
        # where the keys are the model names and the values are the pipelines.
        self.logger.info("Preparing pipelines for training.")
        self.pipelines = self.get_models_all_targets()

    def get_models_all_targets(self) -> dict[float, dict[str, Any]]:
        """Get the models for all targets.

        Returns:
            dict[str, Any]: A dictionary containing the model names as keys and their corresponding pipelines as values.
        """
        out_pipelines = {}
        # Altitude targets and temp or density as targets
        for alt in self.target_arr.keys():
            if alt not in out_pipelines.keys():
                out_pipelines[alt] = {}
            # Now, temp or density
            for target in self.target_arr[alt].keys():
                # Create a new config for the model with the target
                model_config = self.config.model_copy()
                # Create pipelines for each model in the config
                pipelines = pipeline_factory(model_config)
                out_pipelines[alt][target] = pipelines
        return out_pipelines

    def get_targets(
        self, target_idx: np.ndarray
    ) -> dict[float, dict[str, list[float]]]:
        """Get the targets for the training data.

        Args:
            target_idx (np.ndarray): The indices of the target profiles to use.

        Returns:
            dict[float, dict[str, list[float]]]: Dict containing the temperature and density profiles for each altitude.
        """
        # Get the density profile indices from the target_idx
        output_targets: dict[float, dict[str, list[float]]] = {
            alt: {"temp": [], "density": []} for alt in self.atmo_prof["0"]["altitude"]
        }
        # Iterate over the target indices and fill the output_targets dictionary
        for alt in output_targets.keys():
            for idx in target_idx:
                # Get the density and temperature for the current altitude
                output_targets[alt]["temp"].append(
                    self.atmo_prof[str(idx)]["temp"][
                        self.atmo_prof[str(idx)]["altitude"].index(alt)
                    ]
                )
                output_targets[alt]["density"].append(
                    self.atmo_prof[str(idx)]["density"][
                        self.atmo_prof[str(idx)]["altitude"].index(alt)
                    ]
                )
        return output_targets

    def train_test_split(self, train_split: float) -> tuple[np.ndarray, np.ndarray]:
        """Split the aggregated data into training and testing sets.

        Args:
            train_split (float): The fraction of the data to use for training.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the training data and testing data.
        """
        # Get a list of random indices to shuffle the data
        if self.config.train_params.shuffle:
            # If shuffle is True, we shuffle the data
            indices = np.random.default_rng().choice(
                len(self.agg_data), len(self.agg_data), replace=False
            )
        else:
            # If shuffle is False, we just create a range of indices
            indices = np.arange(len(self.agg_data))
        # Now, take the first `train_split` fraction of the indices for training
        split_idx = int(len(self.agg_data) * train_split)
        # Split and randomize the data
        train_data = self.agg_data[indices[:split_idx]]
        self.train_idx = self.target_idx[
            indices[:split_idx]
        ]  # Don't forget to shuffle the target indices as well
        test_data = self.agg_data[indices[split_idx:]]
        self.test_idx = self.target_idx[indices[split_idx:]]
        return train_data, test_data

    def train(self) -> dict[float, dict[str, dict[str, Any]]]:
        """Train the models using the aggregated data."""
        # Iterate over the pipelines and fit them to the aggregated data
        # A model is trained for each altitude in the aggregated data
        out_models: dict[float, dict[str, dict[str, Any]]] = {}
        for alt in self.pipelines.keys():
            if alt not in out_models.keys():
                out_models[alt] = {}
            for target in self.pipelines[alt].keys():
                if target not in out_models[alt].keys():
                    out_models[alt][target] = {}
                for model_name, pipeline in self.pipelines[alt][target].items():
                    if model_name not in out_models[alt][target].keys():
                        out_models[alt][target][model_name] = {
                            "score": None,
                            "model": None,
                        }
                    self.logger.info(
                        f"Training model: {model_name} for altitude {alt} and target {target}"
                    )
                    pipeline.fit(self.train_data, self.target_arr[alt][target])
                    best_est = pipeline.best_estimator_
                    # Replace the custom scaler with the sklearn scaler (saved as attribute)
                    if best_est.steps[0][1].option != "none":
                        best_est.steps[0] = ("scaler", best_est.steps[0][1].scaler)
                    else:
                        best_est.steps.pop(0)  # Remove the scaler if it is None
                    self.logger.info(
                        f"Model {model_name} trained with score: {pipeline.best_score_}"
                    )
                    out_models[alt][target][model_name]["score"] = pipeline.best_score_
                    out_models[alt][target][model_name]["model"] = best_est
        # Return the trained models
        return out_models

    def export_models_local(
        self,
        onnx_model: Any,
        model_name: str,
        output_config: SaveLocalConfig,
    ) -> str:
        """Export the trained model to a local directory in ONNX format.

        Args:
            onnx_model (Any): The ONNX model to export.
            model_name (str): The name of the model to export.
            output_config (SaveLocalConfig): Configuration needed to save the model.

        Returns:
            str: The hash of the exported model.
        """
        serialized_model = onnx_model.SerializeToString()
        # Calculate the hash of the model
        model_hash = hashlib.sha256(serialized_model).hexdigest()
        with open(output_config.save_dir / model_name, "wb") as f:
            f.write(serialized_model)  # type: ignore
        return model_hash

    def export_models_http(
        self,
        onnx_model: Any,
        model_name: str,
        output_config: SaveHttpConfig,
    ) -> str:
        """Export the trained model to a HTTP server.

        Args:
            onnx_model (Any): The ONNX model to export.
            model_name (str): The name of the model to export.
            output_config (SaveHttpConfig): Configuration needed to save the model.

        Returns:
            str: The hash of the exported model.
        """

        serialized_model = onnx_model.SerializeToString()
        # Calculate the hash of the model
        model_hash = hashlib.sha256(serialized_model).hexdigest()
        # Prepare the URL for the HTTP request
        url = (
            output_config.url + model_name
            if output_config.url.endswith("/")
            else output_config.url + "/" + model_name
        )
        # Send the model to the HTTP server
        response = requests.put(
            url,
            data=serialized_model,
            headers={
                "Content-Type": "application/octet-stream",
                "Authorization": f"Bearer {output_config.write_token}",
            },
        )
        if not response.ok:
            self.logger.error(
                f"Error exporting model {model_name} to HTTP server: {response.text}"
            )
            raise Exception(f"Failed to export model {model_name} to HTTP server.")
        return model_hash

    def export_models(
        self,
        models: dict[float, dict[str, dict[str, Any]]],
        output_config: SaveLocalConfig | SaveHttpConfig,
    ) -> None:
        """Export the trained models to ONNX format.

        Args:
            models (dict[float, dict[str, dict[str, Any]]]): The trained models to export.
            output_config (SaveLocalConfig | SaveHttpConfig): Configuration needed to save the model.
        """
        self.logger.info("Exporting models to ONNX format...")
        if output_config.type == "local":
            # If the output_dir is a local directory, we create it if it does not exist
            output_dir = output_config.save_dir
            # Create the directory if it does not exist
            output_dir.mkdir(parents=True, exist_ok=True)
            export_func = self.export_models_local
        else:
            output_dir = output_config.url
            export_func = self.export_models_http

        scores: dict[str, dict[str, dict[str, list[float | str]]]] = {}
        for alt in models.keys():  # Altitudes
            scores[str(alt)] = {}
            for target in models[alt].keys():  # Targets (temp or density)
                scores[str(alt)][target] = {"scores": [], "models": [], "hash": []}
                for model_name, model_info in models[alt][target].items():
                    # Convert the model to ONNX format
                    try:
                        onnx_model = to_onnx(
                            model_info["model"],
                            self.train_data.astype(np.float32),
                        )
                        # Save the ONNX model
                        model_name = str(uuid.uuid4()) + ".onnx"
                        hash = export_func(onnx_model, model_name, output_config)  # type: ignore
                        self.logger.info(
                            f"Model {model_name} for altitude {alt} and target {target} exported to {output_config.type}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error converting model {model_name} for altitude {alt} and target {target} to ONNX: {e}"
                        )
                        self.logger.error("Skipping model.")
                        # If there is an error, we log it and continue with the next model
                        continue
                    # Store the score
                    scores[str(alt)][target][f"scores"].append(model_info["score"])
                    # The path to the ONNX model
                    scores[str(alt)][target][f"models"].append(model_name)
                    # And the hash of the model
                    scores[str(alt)][target][f"hash"].append(hash)
        # Save the scores to a JSON file
        output: dict[str, Any] = {"scorer_function": self.config.train_params.refit}
        output["scores"] = scores
        # Check if the scores.json file already exists in the server or local directory
        if output_config.type == "local":
            scores_path = (
                output_dir / "scores.json"
                if isinstance(output_dir, pathlib.Path)
                else pathlib.Path(output_dir) / "scores.json"
            )
            if scores_path.exists():
                # Read the json file
                with open(scores_path, "r") as f:
                    existing_scores = json.load(f)
                output = self.update_score_dict(existing_scores, output)
            # Write the updated scores to the JSON file
            with open(scores_path, "w") as f:
                json.dump(output, f, indent=4)
            # And the config used to train the models
            config_path = (
                output_dir / "config.json"
                if isinstance(output_dir, pathlib.Path)
                else pathlib.Path(output_dir) / "config.json"
            )
            with open(config_path, "w") as f:
                f.write(self.config.model_dump_json(indent=4))
        else:
            # If the output is a HTTP server we have to GET the scores.json file if it exists
            score_path = (
                output_config.url + "scores.json"
                if output_config.url.endswith("/")
                else output_config.url + "/scores.json"
            )
            score_files = requests.get(
                score_path,
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {output_config.read_token}",
                },
            )
            if score_files.ok:
                # If the file exists, we load it
                existing_scores = score_files.json()
                # Update the existing scores with the new ones
                output = self.update_score_dict(existing_scores, output)
            # Post the scores to the HTTP server
            response = requests.put(
                score_path,
                json=output,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {output_config.write_token}",
                },
            )
            if not response.ok:
                self.logger.error(
                    f"Error exporting scores to HTTP server: {response.text}"
                )
                raise Exception("Failed to export scores to HTTP server.")

    def update_score_dict(
        self, original_data: dict[str, Any], new_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Update the scores dictionary in the JSON file with new data.

        Args:
            original_data (pathlib.Path): Original data loaded from the JSON file.
            new_data (dict[str, Any]): New data to update the scores with.

        Returns:
            dict[str, Any]: Updated original data with new scores.
        Raises:
            KeyError: If the "scores" key is not present in the original data.
        """
        # We have to update the "scores" key in the original_data with the new_data
        if "scores" not in original_data:
            raise KeyError(
                "The original data does not contain the 'scores' key. Cannot update scores."
            )
        # Iterate over the new data and update the original data
        for alt, targets in new_data["scores"].items():
            if alt not in original_data["scores"]:
                original_data["scores"][alt] = {}
            for target, models in targets.items():
                if target not in original_data["scores"][alt]:
                    original_data["scores"][alt][target] = {
                        "scores": [],
                        "models": [],
                        "hash": [],
                    }
                for name, score_list in models.items():
                    # Update the scores and models
                    original_data["scores"][alt][target][name].extend(score_list)
        return original_data

    def parse_density_prof(
        self, density_profile: str | pathlib.Path
    ) -> dict[str, dict[str, list]]:
        """Parse the density profile file.

        Args:
            density_profile (str | pathlib.Path): Path to the density profile file in JSON format.
        Returns:
            dict[str, dict[str, list]]: A dictionary containing the altitude, temperature, and density data.
        """
        input_density = load_config(density_profile)
        # Each entry contains a list with three elements: [altitude, temperature[k], density[kg/m^3]]
        output_dat = {
            x: {n: [] for n in ["altitude", "density", "temp"]}
            for x in input_density.keys()
        }
        for key, value in input_density.items():
            for entry in value:
                output_dat[key]["altitude"].append(entry[0])
                output_dat[key]["temp"].append(entry[1])
                output_dat[key]["density"].append(entry[2])
        return output_dat
