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

import os
import argparse
import glob
import pandas as pd
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

from nvblox_evaluation.evaluation_utils.parse_nvblox_timing import get_timings_as_dataframe


rcParams.update({'figure.autolayout': True})


# Get the timings for a single run
def get_total_times(filepath: str) -> pd.Series:
    timings = get_timings_as_dataframe(filepath)
    timings = timings[np.logical_not(timings.index.str.startswith('3dmatch'))]
    timings = timings[np.logical_not(
        timings.index.str.startswith('file_loading'))]
    timings = timings[np.logical_not(
        timings.index.str.endswith('stbi'))]
    timings = timings[np.logical_not(timings.index.str.startswith('esdf'))]
    return timings["total_time"]


# Get timings for multiple runs and average them
def get_branch_mean_total_times(timing_dir: str) -> Tuple[pd.Series, np.ndarray]:
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

    # Check that the branch results are in the dir
    result_dirs = [name for name in os.listdir(timing_root_dir)
                   if os.path.isdir(os.path.join(timing_root_dir, name))]
    assert len(
        result_dirs) == 2, "We expect a comparision between exactly two branches"

    # Branch 1 results
    branch_1_name = result_dirs[0]
    branch_1_results_folder = os.path.join(timing_root_dir, branch_1_name)
    branch_1_mean, branch_1_err = get_branch_mean_total_times(
        branch_1_results_folder)

    # Branch 2 results
    branch_2_name = result_dirs[1]
    branch_2_results_folder = os.path.join(timing_root_dir, branch_2_name)
    branch_2_mean, branch_2_err = get_branch_mean_total_times(
        branch_2_results_folder)

    # Select only the intersection of the two series.
    intersection_columns = branch_1_mean.index.intersection(
        branch_2_mean.index)

    branch_1_inds = [branch_1_mean.index.get_loc(
        c) for c in intersection_columns]
    branch_2_inds = [branch_2_mean.index.get_loc(
        c) for c in intersection_columns]

    branch_1_mean_filtered = branch_1_mean[intersection_columns]
    branch_2_mean_filtered = branch_2_mean[intersection_columns]
    branch_1_err_filtered = branch_1_err[:, branch_1_inds]
    branch_2_err_filtered = branch_2_err[:, branch_2_inds]

    assert len(branch_2_mean_filtered) == len(branch_1_mean_filtered)

    # Plot
    fig, ax = plt.subplots()
    bar_width = 0.35
    x = np.arange(len(branch_1_mean_filtered))
    x_device = x - bar_width / 2.0
    x_unified = x + bar_width / 2.0
    bar1 = ax.bar(x_device, branch_1_mean_filtered, bar_width,
                  yerr=branch_1_err_filtered, label=branch_1_name)
    bar2 = ax.bar(x_unified, branch_2_mean_filtered, bar_width,
                  yerr=branch_2_err_filtered, label=branch_2_name)
    try:
        ax.bar_label(bar1, fmt='%.2f')
        ax.bar_label(bar2, fmt='%.2f')
    except:
        pass
    ax.set_xticks(x)
    ax.set_xticklabels(intersection_columns, rotation='vertical')
    ax.legend()

    # Save the plot in the root folder of the timings.
    if save_fig:
        image_path = os.path.join(timing_root_dir, 'timings.png')
        print("Saving figure to disk as: " + image_path)
        plt.savefig(image_path)

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot branch timing comparisons.")
    parser.add_argument("timing_root_dir", metavar="timing_root_dir", type=str,
                        help="The directory containing the timing results.")
    args = parser.parse_args()
    plot_timings(args.timing_root_dir)
