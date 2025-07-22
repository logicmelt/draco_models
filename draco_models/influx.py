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
    ) -> Any:
        """
        Execute a custom query against the InfluxDB instance.

        Args:
            query (str): The InfluxQL or Flux query to execute.

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
