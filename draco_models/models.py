import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from typing import Any


class MovingAverage:
    def __init__(self, window_size: int) -> None:
        self.window_size = window_size
        self.data = []
        self.last_average = 0.0

    def add_data(self, value: float) -> None:
        """Adds a new value to the moving average calculation."""
        # Store the last average before updating
        self.last_average = self.get_average()
        # Update the data with the new value
        self.data.append(value)
        if len(self.data) > self.window_size:
            self.data.pop(0)

    def add_array(self, values: np.ndarray) -> None:
        """Adds an array of values to the moving average calculation."""
        if len(values) > self.window_size:
            assert False, "Array length exceeds window size."
        for value in values:
            self.add_data(value)

    def get_average(self) -> float:
        """Returns the current moving average."""
        if not self.data:
            return 0.0
        return np.mean(self.data).item()

    def get_flag(self, threshold: float) -> bool:
        """Returns a binary flag. True if the relative change between the current moving average and the previous
        is larger than the threhsold. False otherwise."""
        if not self.data:
            return False
        current_average = self.get_average()
        if self.last_average == 0:
            return False
        relative_change = abs((current_average - self.last_average) / self.last_average)
        print(relative_change, current_average, self.last_average)
        return relative_change > threshold

    def reset(self) -> None:
        """Resets the moving average data."""
        self.data = []


class SVRModel:
    def __init__(self, **kwargs: Any) -> None:
        """
        Initializes the SVR model with optional parameters.

        Args:
            **kwargs: Optional parameters for the scikit SVR model
        """
        self.scaler = StandardScaler()
        self.model = SVR(**kwargs)
        self.is_fitted = False

    def fit_scaler(self, x: np.ndarray) -> None:
        """
        Fits the scaler to the input data.

        Args:
            x (np.ndarray): Input data to fit the scaler.
        """
        self.scaler.fit(x)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Fits the SVR model to the input data.

        Args:
            x (np.ndarray): Input features.
            y (np.ndarray): Target values.
        """
        if not self.is_fitted:
            self.fit_scaler(x)
            x_scaled = self.scaler.transform(x)
            self.model.fit(x_scaled, y)
            self.is_fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the input data.

        Args:
            x (np.ndarray): Input features.

        Returns:
            np.ndarray: Predicted target values.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call 'fit' before 'predict'.")
        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled)
