from nvblox.experiments.timing import get_timings_as_dataframe
import os
import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})


def plot_timings(timing_root_dir: str) -> None:

    print("Loading result files at: " + timing_root_dir)
    files = glob.glob(os.path.join(timing_root_dir, '*timings.txt'))
    files.sort()
    files.insert(0, files.pop())

    total_times = []
    labels = []
    for filepath in files:
        print("Loading a file at: " + filepath)
        timing_dataframe = get_timings_as_dataframe(filepath)
        total_time_series = timing_dataframe['total_time']
        total_time = total_time_series.at['3dmatch/file_loading']
        total_times.append(total_time)
        labels.append(filepath.split('/')[-1].split('.')[0])

    # Plot
    fig, ax = plt.subplots()
    x = np.arange(len(total_times))
    ax.bar(x, total_times)
    ax.set_ylabel("Load time (s)")
    ax.set_xlabel("Test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical')

    fig_path = os.path.join(timing_root_dir, 'timings.png')
    plt.savefig(fig_path)

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Plot branch timing comparisons.")
    parser.add_argument("timing_root_dir", metavar="timing_root_dir", type=str,
                        help="The directory containing the timing results.")
    args = parser.parse_args()
    plot_timings(args.timing_root_dir)
