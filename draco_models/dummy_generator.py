import numpy as np
import json
from collections import defaultdict

class DummyDataset:
    def __init__(self, density_profile: str = "") -> None:
        self.random_gen = np.random.default_rng()
        self.density_profile = density_profile

    def generate_dataset(
        self, muon_rate: int, muon_std: float, data_len: int
    ) -> np.ndarray:
        """Generates data_len muon readings from a normal distribution
        and rounds them to the closest integer.

        Args:
            data_len (int): Lenght of the sample

        Returns:
            np.ndarray: Generated muon readings
        """
        # Generate data_len simulated muon readings
        muon_read = self.random_gen.normal(muon_rate, muon_std, data_len)
        # Round to integers
        muon_read = muon_read.round(0).astype(int)

        return muon_read

    def generate_anomalous_dataset(
        self,
        muon_rate: int,
        muon_std: float,
        reduction_factor: float,
        data_len: int,
        anomalous_len: int,
    ) -> np.ndarray:

        # The muon rate and std are modulated by the reduction factor
        muon_anom = int(muon_rate * (1 - reduction_factor))
        muon_std_anom = muon_std # * (1 - reduction_factor)
        # The anomalous event might start between 0 and data_len - anomalous_len
        start_anom = np.random.randint(0, data_len - anomalous_len)

        # First, we generate start_anom normal readings
        normal_read = self.generate_dataset(muon_rate, muon_std, start_anom)
        # Then, we generate anomalous dataset and concatenate
        anom_dat = self.generate_dataset(muon_anom, muon_std_anom, anomalous_len)
        # Concatenate them
        muon_out = np.concatenate((normal_read, anom_dat))
        # If the len is lower than data_len add more normal readings
        if len(muon_out) < data_len:
            add_data = self.generate_dataset(
                muon_rate, muon_std, data_len - len(muon_out)
            )
            muon_out = np.concatenate((muon_out, add_data))

        return muon_out
    
    def generate_dataset_with_profile(
        self,
        muon_rate: int,
        muon_std: float,
        data_len: int,
        reduction_factor: float = 0.,
        anomalous_len: int = 0,
        anomaly_data: bool = False
    ) -> tuple[np.ndarray, dict[float, list[float]]]:
        """Generates an anomalous dataset with a density profile."""
        # Generate the dataset
        if anomaly_data: # If anomaly_data is True, generate anomalous data
            muon_data = self.generate_anomalous_dataset(
                muon_rate, muon_std, reduction_factor, data_len, anomalous_len
            )
        else: # Otherwise, generate normal data
            muon_data = self.generate_dataset(muon_rate, muon_std, data_len)
        
        # If a density profile is specified read it and use it
        temp_arr = defaultdict(list)
        if self.density_profile:
            with open(self.density_profile, 'r') as file:
                density_data = json.load(file)
            # Extract the density values and altitudes
            for k in density_data.keys():
                day_data = density_data[k]
                for item in day_data:
                    alt = item[0]
                    temp_arr[alt].append(item[1])
            # Last, interpolate the density values to match the muon data length
            for alt, values in temp_arr.items():
                temp_arr[alt] = self.interpolate(values, np.arange(data_len).tolist())
        
        return muon_data, temp_arr

    def interpolate(self, x: list[float], new_points: list[float]) -> list[float]:
        """Interpolate the given x values to new points."""
        # Get the maximum value of new_points to ensure we don't go out of bounds
        max_point = max(new_points)
        # Get len(x) points up to max_point
        old_points = np.linspace(0, max_point, len(x))
        # Use numpy for interpolation
        return np.interp(new_points, old_points, x).tolist()
