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

import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

from nvblox.experiments.timing import get_timings_as_dataframe


def get_relavent_timings(filepath: str):
    timings = get_timings_as_dataframe(filepath)
    timings = timings.drop(labels="3dmatch/time_per_frame")
    timings = timings.drop(labels="3dmatch/file_loading")
    return timings["total_time"]


def main():
    # Load the data
    input_root = "./output"

    # Averaging the experiment runs
    series = []
    device_timing_files = glob.glob(input_root + "/device_*.txt")
    for f in device_timing_files:
        series.append(get_relavent_timings(f))
    device_total_times = pd.concat(series, axis=1)
    device_total_times_mean = device_total_times.mean(axis=1)
    device_total_times_std = device_total_times.std(axis=1)

    series = []
    unified_timing_files = glob.glob(input_root + "/unified_*.txt")
    for f in unified_timing_files:
        series.append(get_relavent_timings(f))
    unified_total_times = pd.concat(series, axis=1)
    unified_total_times_mean = unified_total_times.mean(axis=1)
    unified_total_times_std = unified_total_times.std(axis=1)

    # Plot
    fig, ax = plt.subplots()
    bar_width = 0.35
    x = np.arange(len(device_total_times_mean))
    x_device = x - bar_width / 2.0
    x_unified = x + bar_width / 2.0
    ax.bar(x_device, device_total_times_mean, bar_width, yerr=device_total_times_std, label='device')
    ax.bar(x_unified, unified_total_times_mean, bar_width, yerr=unified_total_times_std, label='unified')
    ax.set_xticks(x)
    ax.set_xticklabels(device_total_times_mean.index, rotation='vertical')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
