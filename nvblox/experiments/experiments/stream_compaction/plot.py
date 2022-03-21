
from nvblox.experiments.timing import get_timings_as_dataframe
import os
import argparse
import glob
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


# Get the timings for a single run
def get_total_times(filepath: str) -> pd.Series:
    timings = get_timings_as_dataframe(filepath)
    return timings["total_time"]


def get_platform_timings(timings_dir: str, platform_name: str = None) -> pd.DataFrame:
    results_files = glob.glob(timings_dir + "/timings_*.txt")
    results_files.sort()
    df = pd.DataFrame()
    for f in results_files:

        # Extract datasize
        data_size = int(re.search('timings_(.+?).txt', f).group(1))

        # Total times -> timg/byte
        timings = get_total_times(f)
        timings /= data_size

        # Add to timings for each datasize
        this_df = timings.to_frame(name=str(data_size) + " bytes")
        if df.empty:
            df = this_df
        else:
            df = pd.merge(df, this_df, left_index=True, right_index=True)
    if platform_name is not None:
        df = df.rename(index={'host_compaction': platform_name + ' host'})
        df = df.rename(index={'device_compaction': platform_name + ' device'})
    print(df)
    return df


def plot_timings(timings_dir: str):

    df = pd.DataFrame()
    platform_timings = []
    for platform in os.listdir(timings_dir):
        print("collecting timings for platform: " + platform)

        platform_timings.append(get_platform_timings(
            os.path.join(timings_dir, platform), platform))
    df = pd.concat(platform_timings)

    # Plotting
    df.plot.bar()
    plt.yscale('log')
    plt.ylabel('Time per byte (s)')
    plt.xlabel('Platform/method')
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot stream compaction timing comparisons.")
    parser.add_argument("timings_dir", metavar="timings_dir", type=str,
                        help="The directory containing the timing results.")
    args = parser.parse_args()
    plot_timings(args.timings_dir)
