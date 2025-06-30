from draco_models.config import InputConfig, load_config
from draco_models.influx import InfluxDB
from draco_models.aggregator import Aggregator, IdentityAggregator
from draco_models.models import pipeline_factory
from draco_models.utils import create_logger
from typing import Any
from skl2onnx import to_onnx
import pathlib
import numpy as np
import json

# TODO: Add different scalers to the pipelines


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
        self.logger = create_logger(
            "draco_trainer",
            self.config.save_dir / "train_log.job",
            self.config.logging_level,
        )
        # Set up the job.
        # Get the data from InfluxDB using the provided query and columns.
        self.influxdb = InfluxDB(config.influxdb)
        self.logger.info("Fetching data from InfluxDB...")
        self.data = self.influxdb.custom_query(
            config.query,
            config.influxdb.columns_in,
            config.influxdb.columns_out,
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
                    best_score = best_est.score(
                        self.test_data, self.test_arr[alt][target]
                    )

                    out_models[alt][target][model_name]["score"] = best_score
                    out_models[alt][target][model_name]["model"] = best_est

        # Return the trained models
        return out_models

    def export_models(
        self,
        models: dict[float, dict[str, dict[str, Any]]],
        output_dir: pathlib.Path | str,
    ) -> None:
        """Export the trained models to ONNX format.

        Args:
            models (dict[float, dict[str, dict[str, Any]]]): The trained models to export.
            output_dir (pathlib.Path): The directory where the models will be saved.
        """
        self.logger.info("Exporting models to ONNX format...")
        output_dir = (
            pathlib.Path(output_dir) if isinstance(output_dir, str) else output_dir
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        scores: dict[str, dict[str, dict[str, float]]] = {}
        for alt in models.keys():  # Altitudes
            scores[str(alt)] = {}
            for target in models[alt].keys():  # Targets (temp or density)
                scores[str(alt)][target] = {}
                for model_name, model_info in models[alt][target].items():
                    # Convert the model to ONNX format
                    try:
                        onnx_model = to_onnx(
                            model_info["model"],
                            self.train_data.astype(np.float32),
                        )
                        # Save the ONNX model
                        output_path = output_dir / f"{model_name}_{alt}_{target}.onnx"
                        with open(output_path, "wb") as f:
                            f.write(onnx_model.SerializeToString())  # type: ignore
                        self.logger.info(
                            f"Model {model_name} for altitude {alt} and target {target} exported to {output_path}"
                        )
                    except Exception as e:
                        self.logger.error(
                            f"Error converting model {model_name} for altitude {alt} and target {target} to ONNX: {e}"
                        )
                        self.logger.error("Skipping model.")
                        # If there is an error, we log it and continue with the next model
                        continue
                    # Store the score
                    scores[str(alt)][target][f"{model_name}_{alt}_{target}"] = (
                        model_info["score"]
                    )
        # Save the scores to a JSON file
        scores_path = output_dir / "scores.json"
        output: dict[str, Any] = {"scorer_function": self.config.train_params.refit}
        output["scores"] = scores
        with open(scores_path, "w") as f:
            json.dump(output, f, indent=4)
        # And the config used to train the models
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            f.write(self.config.model_dump_json(indent=4))

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
