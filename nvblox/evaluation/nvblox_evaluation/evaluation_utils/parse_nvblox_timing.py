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
from typing import Union
import pandas as pd
import re
import json

def save_timing_statistics(timing_path: Union[str, PathLike], output_directory: Union[str, PathLike]) -> None:
    # Extract the means of the timers
    timings_df = get_timings_as_dataframe(timing_path)
    means_series = timings_df['mean']
    means_series.index = ['mean/' + row_name for row_name in means_series.index]
    total_series = timings_df['total_time']
    total_series.index = ['total/' + row_name for row_name in total_series.index]

    # Write the results to a JSON
    output_timings_path = str(output_directory) + '/timing.json'
    print(f"Writing the timings to: {output_timings_path}")
    with open(output_timings_path, "w") as timings_file:
        json.dump(pd.concat([means_series, total_series]).to_dict(), timings_file, indent=4)


def get_timings_as_dataframe(filepath: Union[str, PathLike]) -> pd.DataFrame:
    """Opens a file containing a nvblox timer dump and returns the data as a pandas Dataframe

    Args:
        filepath (Union[str, Path]): path to file containing the timer string

    Returns:
        pd.DataFrame: A dataframe containing the timing information
    """
    names = []
    num_calls = []
    total_times = []
    means = []
    stds = []
    mins = []
    maxes = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        lines = lines[:-1]
        for line in lines[4:]:
            entries = re.split('\s+|,', line)
            entries = [entry.strip('()[]') for entry in entries]

            names.append(entries[0])
            num_calls.append(int(entries[1]))
            total_times.append(float(entries[2]))
            means.append(float(entries[3]))
            stds.append(float(entries[5]))
            mins.append(float(entries[6]))
            maxes.append(float(entries[7]))

    d = {"num_calls": num_calls, "total_time": total_times,
         "mean": means, "std": stds, "min": mins, "max": maxes}
    return pd.DataFrame(d, index=names)
