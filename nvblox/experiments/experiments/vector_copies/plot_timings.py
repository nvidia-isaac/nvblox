#!/usr/bin/python3

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


def get_total_times(filepath: str) -> pd.Series:
    timings = get_timings_as_dataframe(filepath)
    return timings["total_time"]


def plot_timings(timing_root_dir: str, save_fig: bool = True) -> None:
    print("Looking for results in: " + timing_root_dir)

    # Check that the profile results are in the dir
    result_files = [name for name in os.listdir(timing_root_dir)
                    if os.path.isfile(os.path.join(timing_root_dir, name))]

    # Load every text file in the results dir.
    df = pd.DataFrame()
    for result_name in result_files:
        results_path = os.path.join(timing_root_dir, result_name)
        if os.path.splitext(result_name)[1] != ".txt":
            continue
        timings = get_total_times(results_path)
        this_df = timings.to_frame(name=os.path.splitext(result_name)[0])
        if df.empty:
            df = this_df
        else:
            df = pd.merge(df, this_df, left_index=True, right_index=True)

    print(df)

    print(df.loc["exp/initialize"])

    # Organize meaningful columns
    row_names = ["host to device", "device to device",
                 "device to host", "kernel"]
    df.loc[row_names[0]] = df.loc["exp2a/cuda/host_to_device"]
    df.loc[row_names[1]] = df.loc["exp2b/cuda/device_to_device"]
    df.loc[row_names[2]] = df.loc["exp2c/cuda/device_to_host"]
    df.loc[row_names[3]] = df.loc["exp2a/kernel"]

    # Plot
    ax = df.loc[row_names].plot.bar(rot=0)
    ax.set_ylabel('Total Time (s)')
    for con in ax.containers:
        ax.bar_label(con, fmt='%.2f')
    plt.title("Vector Copy Times Across Platforms")

    # Save the plot in the root folder of the timings.
    if save_fig:
        image_path = os.path.join(timing_root_dir, 'timings.png')
        print("Saving figure to disk as: " + image_path)
        plt.savefig(image_path)
    
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot vector copy timing comparisons.")
    parser.add_argument("timing_root_dir", metavar="timing_root_dir", type=str,
                        help="The directory containing the timing results.")
    args = parser.parse_args()
    plot_timings(args.timing_root_dir)
