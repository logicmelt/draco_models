from typing import Any
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import influxdb_client
import influxdb_client.client.flux_table

from draco_models.config import InfluxDBConfig


# InfluxDB is not typed correctly https://github.com/influxdata/influxdb-client-python/issues/694
class InfluxDB:
    def __init__(self, config: InfluxDBConfig):
        """
        Initialize the InfluxDB client.

        Args:
            config (InfluxDBConfig): Configuration for connecting to the InfluxDB instance.
        """
        self.client = influxdb_client.InfluxDBClient(
            url=config.url, token=config.token, org=config.org
        )  # type: ignore
        self.org = config.org
        self.query_apy = self.client.query_api()

    def custom_query(
        self,
        query: str,
        columns_in: list[str] | str = "",
        columns_out: list[str] | str = "",
    ) -> Any:
        """
        Execute a custom query against the InfluxDB instance.

        Args:
            query (str): The InfluxQL or Flux query to execute.
            columns_in (list[str]): List of columns to include in the parsed output. If empty, all columns are included.
            columns_out (list[str]): List of columns to exclude from the parsed output. If empty, no columns are excluded.

        Returns:
            Any: The result of the query.
        """
        query_out = self.query_apy.query(query=query, org=self.org)
        parsed_query = self.parse_data(query_out, columns_in, columns_out)
        return parsed_query

    def custom_query2(
        self,
        query: str,
        columns_in: list[str] | str = "",
        columns_out: list[str] | str = "",
    ) -> Any:
        """
        Execute a custom query against the InfluxDB instance.

        Args:
            query (str): The InfluxQL or Flux query to execute.
            columns_in (list[str]): List of columns to include in the parsed output. If empty, all columns are included.
            columns_out (list[str]): List of columns to exclude from the parsed output. If empty, no columns are excluded.

        Returns:
            Any: The result of the query.
        """

        result = self.query_apy.query_raw(query, org=self.org)

        df = pd.read_csv(result, skiprows=[0, 1, 2])
        df.drop(
            [
                "Unnamed: 0",
                "result",
                "table",
                "_start",
                "_stop",
                "_measurement",
            ],
            axis=1,
            inplace=True,
        )  # customize as needed
        df["timestamp"] = df["_time"]
        df.set_index("_time", inplace=True)
        df.sort_index(inplace=True)

        output = {}
        for c in df.columns:
            if c in [
                "EventID",
                "Particle",
                "ParticleID",
                "TrackID",
                "density_day_idx",
                "latitude",
                "local_time",
                "longitude",
                "phi",
                "process_ID",
                "px",
                "py",
                "pz",
                "start_time",
                "theta",
                "x",
                "y",
                "z",
                "timestamps",
            ]:
                output[c] = list(df[c])

        timestamps = [datetime.fromisoformat(t).timestamp() for t in df["timestamp"]]
        output["timestamps"] = timestamps

        return output

    def parse_data(
        self,
        input_dat: influxdb_client.client.flux_table.TableList,
        columns_in: list[str] | str = "",
        columns_out: list[str] | str = "",
    ) -> Any:
        """
        Parse the output of a query to a structured format.

        Args:
            input_dat (influxdb_client.client.flux_table.TableList): TableList from InfluxDB query that should be parsed.
            columns_in (list[str]): List of columns to include in the parsed output. If empty, all columns are included.
            columns_out (list[str]): List of columns to exclude from the parsed output. If empty, no columns are excluded.

        Returns:
            Any: The parsed query.
        """
        if isinstance(columns_in, str):
            columns_in = [columns_in]
        if isinstance(columns_out, str):
            columns_out = [columns_out]
        output = defaultdict(list)
        timestamps = []
        tags = (
            set()
        )  # To keep track of unique tags so that we have all the unique timestamps
        for k in input_dat:
            for n in k.records:
                if n["_field"] not in columns_in and columns_in[0] != "":
                    continue
                # Exclude columns if specified
                if n["_field"] in columns_out:
                    continue
                if "particle_id" not in n.values.keys():
                    # This key is only present in the real detectors
                    iter_tag = (n["detector_id"], n["detector_type"], n["run_ID"])
                else:
                    iter_tag = (
                        n["detector_id"],
                        n["detector_type"],
                        n["run_ID"],
                        n["particle_id"],
                    )
                # If the tag (or the timestamp) is not in the set
                # then we have a new entry and we add it
                if iter_tag not in tags or n["_time"] not in timestamps:
                    tags.add(iter_tag)
                    # And the timestamp
                    timestamps.append(n["_time"])
                output[n["_field"]].append(n["_value"])
        # Transform the datetime objects to real timestamps
        timestamps_out = [t.timestamp() for t in timestamps]
        output_len = len(output[list(output.keys())[0]]) if len(output) > 0 else 0
        # Sanity check: ensure that the number of timestamps matches the number of values in the output
        assert len(timestamps_out) == output_len, (
            "The number of timestamps does not match the number of values in the output."
        )
        if output_len == 0:
            return {}
        # Sort the timestamps if needed
        if not self.is_sorted(timestamps_out):
            index_sort = np.argsort(timestamps_out)
            timestamps_out = [timestamps_out[i] for i in index_sort]
            # Sort the output data according to the timestamps
            for key in output.keys():
                output[key] = [output[key][i] for i in index_sort]
        # And add it to the output
        output["timestamps"] = timestamps_out
        return output

    def is_sorted(self, data: list[float | int]) -> np.bool:
        """
        Check if the data is sorted in ascending order.

        Args:
            data (list[float | int]): The data to check.

        Returns:
            bool: True if the data is sorted, False otherwise.
        """
        data_as_np = np.array(data)
        return np.all(data_as_np[:-1] <= data_as_np[1:])
