#!/usr/bin/python3

#
# Copyright 2022 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from os import PathLike
from typing import Union, Dict
import pandas as pd
import re
import json
from collections import defaultdict


def save_timing_statistics(timing_path: Union[str, PathLike],
                           output_directory: Union[str, PathLike]) -> None:
    # Extract the means of the timers
    timings_df = get_timings_as_dataframe(timing_path)
    means_series = timings_df['mean']
    means_series.index = [
        'mean/' + row_name for row_name in means_series.index
    ]
    total_series = timings_df['total_time']
    total_series.index = [
        'total/' + row_name for row_name in total_series.index
    ]

    # Write the results to a JSON
    output_timings_path = str(output_directory) + '/timing.json'
    print(f"Writing the timings to: {output_timings_path}")
    with open(output_timings_path, "w") as timings_file:
        json.dump(pd.concat([means_series, total_series]).to_dict(),
                  timings_file,
                  indent=4)


def get_table_as_dataframe_from_string(
        table_string: str,
        name_to_column_index: Dict[str, int],
        start_row: int = 0,
        remove_last_row: bool = True) -> pd.DataFrame:
    """Read an nvblox table from a .txt file and returns it as a DataFrame.

    Args:
        filepath (Union[str, PathLike]): Path to the .txt table.
        name_to_column_index (Dict[str, int]): A map containing column names and their
                                               corresponding column indices.
        start_row (int, optional): The row of the text file where the tabular data starts.
                                   Defaults to 0.
        remove_last_row (bool, optional): Set true to remove the last row of the txt table before
                                          parsing. Useful for removing end delimiters.

    Returns:
        pd.DataFrame: A DataFrame containing the tabular data.

    """
    index = []
    stats = defaultdict(list)
    lines = table_string.splitlines()
    if remove_last_row:
        lines = lines[:-1]
    for line in lines[start_row:]:
        entries = re.split('\s+|,', line)
        entries = [entry.strip('()[]') for entry in entries]
        index.append(entries[0])
        for name, column_idx in name_to_column_index.items():
            stats[name].append(float(entries[column_idx]))
    return pd.DataFrame(stats, index=index)


def get_table_as_dataframe(filepath: Union[str, PathLike],
                           name_to_column_index: Dict[str, int],
                           start_row: int = 0,
                           remove_last_row: bool = True) -> pd.DataFrame:
    """Read an nvblox table from a .txt file and returns it as a DataFrame.

    Args:
        filepath (Union[str, PathLike]): Path to the .txt table.
        name_to_column_index (Dict[str, int]): A map containing column names and their
                                               corresponding column indices.
        start_row (int, optional): The row of the text file where the tabular data starts.
                                   Defaults to 0.
        remove_last_row (bool, optional): Set true to remove the last row of the txt table before
                                          parsing. Useful for removing end delimiters.

    Returns:
        pd.DataFrame: A DataFrame containing the tabular data.

    """
    with open(filepath, 'r') as f:
        file_str = f.read()
        return get_table_as_dataframe_from_string(file_str,
                                                  name_to_column_index,
                                                  start_row, remove_last_row)


def get_timings_as_dataframe(filepath: Union[str, PathLike]) -> pd.DataFrame:
    """Open a file containing a nvblox timer and return the data as a pandas Dataframe.

    Args:
        filepath (Union[str, Path]): path to file containing the timer string

    Returns:
        pd.DataFrame: A dataframe containing the timing information

    """
    name_to_column_index = {
        'num_calls': 1,
        'total_time': 2,
        'mean': 3,
        'std': 5,
        'min': 6,
        'max': 7
    }
    start_row = 4
    return get_table_as_dataframe(filepath,
                                  name_to_column_index,
                                  start_row=start_row)


def get_rates_as_dataframe(filepath: Union[str, PathLike]) -> pd.DataFrame:
    """Open a file containing a nvblox rates and return the data as a pandas DataFrame.

    Args:
        filepath (Union[str, Path]): path to file containing the timer string

    Returns:
        pd.DataFrame: A DataFrame containing the timing information

    """
    name_to_column_index = {'num_samples': 1, 'mean': 2}
    start_row = 4
    return get_table_as_dataframe(filepath,
                                  name_to_column_index,
                                  start_row=start_row)
