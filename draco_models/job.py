from draco_models.config import InputConfig, load_config
from draco_models.influx import InfluxDB
from draco_models.aggregator import Aggregator
from draco_models.models import pipeline_factory
from typing import Any
import pathlib
import numpy as np


class Job:
    """Job class to handle the data processing and model training pipeline."""

    def __init__(self, config: InputConfig):
        """Initialize the Job class.
        Args:
            config (InputConfig): The input configuration containing the parameters for the job.
        """
        self.train_idx: np.ndarray
        self.test_idx: np.ndarray
        self.config = config
        # Set up the job.
        # Get the data from InfluxDB using the provided query and columns.
        self.influxdb = InfluxDB(config.influxdb)
        self.data = self.influxdb.custom_query(
            config.query,
            config.influxdb.columns_in,
            config.influxdb.columns_out,
        )
        # Get the density profile file path
        self.atmo_prof = self.parse_density_prof(config.density_profile)
        # Define the aggregator to process the data
        self.aggregator = Aggregator()
        # Now, we have to aggregate the data in time_resolution steps
        self.agg_data, self.target_idx = self.aggregate_time_resolution(
            self.data, config.time_resolution
        )
        # And now we can split the data into training and testing sets
        self.train_data, self.test_data = self.train_test_split(
            self.config.train_params.train_split
        )
        # Get the targets ready for training
        self.target_arr = self.get_targets(self.train_idx)
        self.test_arr = self.get_targets(self.test_idx)
        # Prepare the pipeline for training. This is a dictionary of pipelines
        # where the keys are the model names and the values are the pipelines.
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
        indices = np.random.default_rng().choice(
            len(self.agg_data), len(self.agg_data), replace=False
        )
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

    def train(self):
        """Train the models using the aggregated data."""
        # Iterate over the pipelines and fit them to the aggregated data
        # A model is trained for each altitude in the aggregated data
        out_models = {}
        predict = {"temp": {}, "density": {}}
        for alt in self.pipelines.keys():
            if alt != 0.24:
                # Skip altitudes that are not 0.2 km
                continue
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
                    print(
                        f"Training model: {model_name} for altitude {alt} and target {target}"
                    )
                    pipeline.fit(self.train_data, self.target_arr[alt][target])

                    out_models[alt][target][model_name]["score"] = pipeline.best_score_
                    out_models[alt][target][model_name][
                        "model"
                    ] = pipeline.best_estimator_
                    predict[target][model_name] = pipeline.best_estimator_.predict(
                        self.test_data
                    )

    def aggregate_time_resolution(
        self, data: dict[str, list], time_resolution: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate the data in time resolution steps.

        Args:
            data (dict[str, list]): The data to be aggregated.
            time_resolution (float): The time resolution in seconds.

        Returns:
            tuple[np.ndarray, np.ndarray]: A numpy array containing the aggregated data and the indices of the density profile used.
        """
        # Each aggregated step will cover time_resolution seconds.
        start_time = data["timestamps"][0]  # Start from the first timestamp
        time_arr = np.array(data["timestamps"])
        output_data = []
        target_idx = []
        total_keys = 2  # Initialize total_keys to 2 to avoid reshape errors
        while start_time < time_arr[-1]:
            end_time = start_time + time_resolution
            # Filter the data for the current time step
            start_idx = np.argmax(start_time <= time_arr)
            end_idx = np.argmin(time_arr < end_time)
            if start_idx == end_idx or end_idx == 0:
                # If no data is available for this time step, skip
                start_time = end_time
                continue
            filtered_data = {
                key: value[start_idx:end_idx]
                for key, value in data.items()
                if len(value[start_idx:end_idx]) > 0
            }
            # Aggregate the filtered data
            aggregated_step = self.aggregator.aggregate(filtered_data)
            total_keys = len(aggregated_step)
            # Store the aggregated step
            agg_list = []
            for key, value in aggregated_step.items():
                if key == "density_day_idx":
                    # if the key is the density_day_idx we store the index separately
                    target_idx.append(value)
                    continue
                agg_list.append(value)
            # Append the aggregated step to the output data
            output_data.append(agg_list)
            # Move to the next time step
            start_time = end_time
        # Convert the output data to a numpy array for easier handling
        # Substract 1 from total_keys to account for the density_day_idx
        output_data = np.array(output_data).reshape(-1, total_keys - 1)
        # Get the density_day_idx as array
        target_idx = np.array(target_idx)
        return output_data, target_idx

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
