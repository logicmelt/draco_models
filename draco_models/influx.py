import influxdb_client
import influxdb_client.client.flux_table
from typing import Any
from collections import defaultdict


# InfluxDB is not typed correctly https://github.com/influxdata/influxdb-client-python/issues/694
class InfluxDB:
    def __init__(self, url: str, token: str, org: str):
        """
        Initialize the InfluxDB client.

        Args:
            url (str): The URL of the InfluxDB instance.
            token (str): The authentication token for the InfluxDB instance.
            org (str): The organization name in InfluxDB.
        """
        self.client = influxdb_client.InfluxDBClient(url=url, token=token, org=org)  # type: ignore
        self.org = org
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
        for k in input_dat:
            for n in k.records:
                if n["_field"] not in columns_in and columns_in[0] != "":
                    continue
                # Exclude columns if specified
                if n["_field"] in columns_out:
                    continue
                if n["_time"] not in timestamps:
                    timestamps.append(n["_time"])
                # timestamps[n["_field"]].append(n["_time"].timestamp())
                output[n["_field"]].append(n["_value"])
        # Transform the datetime objects to real timestamps
        timestamps_out = [t.timestamp() for t in timestamps]
        # Sanity check: ensure that the number of timestamps matches the number of values in the output
        assert len(timestamps_out) == len(
            output[list(output.keys())[0]]
        ), "The number of timestamps does not match the number of values in the output."
        # And add it to the output
        output["timestamps"] = timestamps_out
        return output
