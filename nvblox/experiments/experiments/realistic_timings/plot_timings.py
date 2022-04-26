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


from nvblox.experiments.timing import get_timings_as_dataframe
import os
import argparse
import glob
import pandas as pd
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

# Get the timings for a single run - ONLY the 3dmatch timings.

frame_rate = 30

def get_total_times(filepath: str) -> pd.Series:
    timings = get_timings_as_dataframe(filepath)
    timings = timings[timings.index.str.startswith('3dmatch')]
    total_frames = timings["num_calls"].max()
    total_seconds = total_frames / frame_rate
    return timings["total_time"].divide(total_seconds)

# Get timings for multiple runs and average them


def get_profile_mean_total_times(timing_dir: str) -> Tuple[pd.Series, np.ndarray]:
    series = []
    results_files = glob.glob(timing_dir + "/run_*.txt")
    for f in results_files:
        series.append(get_total_times(f))
    total_times = pd.concat(series, axis=1)
    total_times_mean = total_times.mean(axis=1)
    total_times_max = total_times.max(axis=1)
    total_times_min = total_times.min(axis=1)
    total_times_max_min_np = np.vstack((total_times_max, total_times_min))
    total_times_err = np.abs(total_times_max_min_np -
                             total_times_mean.to_numpy())
    return total_times_mean, total_times_err


def plot_timings(timing_root_dir: str, save_fig: bool = True) -> None:
    print("Looking for results in: " + timing_root_dir)

    # Check that the profile results are in the dir
    result_dirs = [name for name in os.listdir(timing_root_dir)
                   if os.path.isdir(os.path.join(timing_root_dir, name))]
    assert len(
        result_dirs) == 2, "We expect a comparision between exactly two profiles (out of laziness)"

    # Profile 1 results
    profile_1_name = result_dirs[0]
    profile_1_results_folder = os.path.join(timing_root_dir, profile_1_name)
    profile_1_mean, profile_1_err = get_profile_mean_total_times(
        profile_1_results_folder)

    # Profile 2 results
    profile_2_name = result_dirs[1]
    profile_2_results_folder = os.path.join(timing_root_dir, profile_2_name)
    profile_2_mean, profile_2_err = get_profile_mean_total_times(
        profile_2_results_folder)

    # Select only the intersection of the two series.
    intersection_columns = profile_1_mean.index.intersection(
        profile_2_mean.index)

    profile_1_inds = [profile_1_mean.index.get_loc(
        c) for c in intersection_columns]
    profile_2_inds = [profile_2_mean.index.get_loc(
        c) for c in intersection_columns]

    profile_1_mean_filtered = profile_1_mean[intersection_columns]
    profile_2_mean_filtered = profile_2_mean[intersection_columns]
    profile_1_err_filtered = profile_1_err[:, profile_1_inds]
    profile_2_err_filtered = profile_2_err[:, profile_2_inds]

    assert len(profile_2_mean_filtered) == len(profile_1_mean_filtered)

    # Plot
    fig, ax = plt.subplots()
    bar_width = 0.35
    x = np.arange(len(profile_1_mean))
    x_device = x - bar_width / 2.0
    x_unified = x + bar_width / 2.0
    bar1 = ax.bar(x_device, profile_1_mean_filtered, bar_width,
           yerr=profile_1_err_filtered, label=profile_1_name)
    bar2 = ax.bar(x_unified, profile_2_mean_filtered, bar_width,
           yerr=profile_2_err_filtered, label=profile_2_name)
    ax.set_xticks(x)
    ax.set_xticklabels(intersection_columns, rotation='vertical')
    try:
        ax.bar_label(bar1, fmt='%.2f')
        ax.bar_label(bar2, fmt='%.2f')
    except:
        pass
    ax.legend()
    ax.set_ylabel('Processing Time per Second of Data (s)')

    # Save the plot in the root folder of the timings.
    if save_fig:
        image_path = os.path.join(timing_root_dir, 'timings.png')
        print("Saving figure to disk as: " + image_path)
        plt.savefig(image_path)

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot profile timing comparisons.")
    parser.add_argument("timing_root_dir", metavar="timing_root_dir", type=str,
                        help="The directory containing the timing results.")
    args = parser.parse_args()
    plot_timings(args.timing_root_dir)
