import scipy.stats
import numpy as np
from typing import Any
from collections import Counter


class IdentityAggregator(object):
    """A class that does not aggregate the data but returns it as is."""

    def __init__(self):
        """
        Initialize the IdentityAggregator class.
        This class will be used to return the data as is without any aggregation.
        """
        super().__init__()
        # Relative order of the data in the output array.
        self.data_order: dict[str, int] = {
            "mean_azimuth": 0,
            "std_azimuth": 1,
            "skewness_azimuth": 2,
            "kurtosis_azimuth": 3,
            "mean_zenith": 4,
            "std_zenith": 5,
            "skewness_zenith": 6,
            "kurtosis_zenith": 7,
            "n_readings": 8,
            # "multiplicity": 9,
        }

    def aggregate_time_resolution(
        self, data: dict[str, list[float | int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Parse the input data to an array so that it can be used for training.

        Args:
            data (dict[str, list[float | int]]): Data to be parsed to an array. It should contain an entry with the density IDx and timestamps.

        Returns:
            tuple[np.ndarray, np.ndarray]: A numpy array containing the parsed data and the indices of the density profile used.
        """
        output_data = np.zeros((len(data["timestamps"]), len(self.data_order)))
        # We will also store the density
        target_idx = np.zeros(len(data["timestamps"]), dtype=int)
        for key, val in data.items():
            if key == "timestamps":
                # If the key is timestamps we skip it
                continue
            if key == "density_day_idx":
                # If the key is density_day_idx we store the index separately
                target_idx = np.array(val, dtype=int) - 1
                continue
            # Store the data in the output array
            output_data[:, self.data_order[key]] = np.array(val)

        return output_data, target_idx


class Aggregator(object):

    def __init__(self, time_resolution: float = 60.0):
        """
        Initialize the Aggregator class.
        This class will be used to aggregate data from an InfluxDB instance covering the desired time range.

        See the documentation for more detailes about the keys in the data dictionary.

        Args:
            time_resolution (float): The time resolution in seconds for aggregating the data. Defaults to 60.0 seconds.
        """
        super().__init__()
        self.time_resolution = time_resolution
        # Relative order of the data in the output array.
        self.data_order: dict[str, int] = {
            "mean_azimuth": 0,
            "std_azimuth": 1,
            "skewness_azimuth": 2,
            "kurtosis_azimuth": 3,
            "mean_zenith": 4,
            "std_zenith": 5,
            "skewness_zenith": 6,
            "kurtosis_zenith": 7,
            "n_readings": 8,
            # "multiplicity": 9,
        }

    def aggregate(self, data: dict[str, Any]) -> dict[str, Any]:
        """Aggregate the data. This method estimates the mean, stdev, kurtosis and skewness of several parameters.

        Args:
            data (dict[str, Any]): Data from an influx database.

        Returns:
            dict[str, Any]: A dictionary containing the average multiplicity and the standard deviation of the multiplicity.
        """
        output_aggregate: dict[str, Any] = {}

        # output_aggregate["multiplicity"] = self.get_multiplicity(data).item()
        # Get the mean, standard deviation, kurtosis and skewness of the azimuthal angle
        output_aggregate["mean_azimuth"] = np.mean(data["phi"]).item()
        output_aggregate["std_azimuth"] = np.std(data["phi"], ddof=1).item()
        output_aggregate["skewness_azimuth"] = scipy.stats.skew(data["phi"]).item()
        output_aggregate["kurtosis_azimuth"] = scipy.stats.kurtosis(data["phi"]).item()

        # Get the mean, standard deviation, kurtosis and skewness of the zenithal angle
        output_aggregate["mean_zenith"] = np.mean(data["theta"]).item()
        output_aggregate["std_zenith"] = np.std(data["theta"], ddof=1).item()
        output_aggregate["skewness_zenith"] = scipy.stats.skew(data["theta"]).item()
        output_aggregate["kurtosis_zenith"] = scipy.stats.kurtosis(data["theta"]).item()

        # Get the total number of readings
        output_aggregate["n_readings"] = len(data["EventID"])

        # And last, the density_day_idx so that we can know which density profile was used
        density_idx = np.unique(data["density_day_idx"]).tolist()
        assert len(density_idx) == 1, (
            "There should be only one density_day_idx in the data. "
            "If you are using more than one density profile, please choose a lower time resolution so that they are separated."
        )
        output_aggregate["density_day_idx"] = density_idx[0]

        return output_aggregate

    def aggregate_time_resolution(
        self, data: dict[str, list[Any]], time_resolution: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate the data in time resolution steps.

        Args:
            data (dict[str, list]): The data to be aggregated.
            time_resolution (float | None): The time resolution in seconds. If None, use the default time resolution of the class.

        Returns:
            tuple[np.ndarray, np.ndarray]: A numpy array containing the aggregated data and the indices of the density profile used.
        """
        time_resolution = (
            self.time_resolution if time_resolution is None else time_resolution
        )
        if len(data) == 0:
            # If the data is empty then the query returned no results
            raise ValueError(
                "The data is empty. The query returned no results. Please check the query and the InfluxDB instance."
            )
        # Each aggregated step will cover time_resolution seconds.
        start_time = data["timestamps"][0]  # Start from the first timestamp
        time_arr = np.array(data["timestamps"])
        output_data = []
        target_idx = []
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
            aggregated_step = self.aggregate(filtered_data)
            # Store the aggregated step
            agg_list = []
            target_idx.append(aggregated_step["density_day_idx"])
            for key in self.data_order.keys():
                agg_list.append(aggregated_step[key])
            # Append the aggregated step to the output data
            output_data.append(agg_list)
            # Move to the next time step
            start_time = time_arr[end_idx]
        # Convert the output data to a numpy array for easier handling
        # Substract 1 from total_keys to account for the density_day_idx
        output_data = np.array(output_data).reshape(-1, len(self.data_order))
        # Get the density_day_idx as array
        target_idx = np.array(target_idx)
        return output_data, target_idx

    def get_multiplicity(self, data: dict[str, Any]) -> np.floating:
        """Get the average multiplicity of the particles.

        Args:
            data (dict[str, Any]): Data from an influx database including "EventID" and "process_ID" keys.

        Returns:
            np.float64: The average multiplicity of the particles.
        """
        # We need two columns: EventID and process_ID
        # If EventID is the same then it was generated by the same primary but we have to check that the process_ID is also the same
        eventid = np.array(data["EventID"])
        processid = np.array(data["process_ID"])
        # Get the unique process_ID
        unique_process = np.unique(processid)
        # Now iterate and get the multiplicity of each process
        multiplicity: list[int] = []
        for process in unique_process:
            # Get the indexes of the process
            indexes = np.where(processid == process)[0]
            iter_multi = Counter(eventid[indexes])
            multiplicity.extend(iter_multi.values())
        # The multiplicity should start from 0 so we have to subtract 1
        multiplicity = [x - 1 for x in multiplicity if x > 0]
        return np.mean(multiplicity)
