#!/usr/bin/python3

import os
import git
import subprocess


# Experiment params
# The list of sizes over which to run the experiments
data_sizes = [1e3, 1e4, 1e5, 1e6, 1e7]


def run() -> None:

    # Create the output path
    # datetime_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    this_directory = os.getcwd()
    output_root = os.path.join(this_directory, 'output', 'platform')
    os.makedirs(output_root, exist_ok=True)

    # Binary path
    repo = git.Repo(this_directory, search_parent_directories=True)
    git_root_dir = repo.git.rev_parse("--show-toplevel")
    assert not repo.bare
    build_dir = os.path.join(git_root_dir, 'nvblox/build')
    binary_path = os.path.join(
        build_dir, 'experiments/experiments/stream_compaction/stream_compaction')

    # Running the experiment for each size
    for data_size in data_sizes:

        output_path = os.path.join(
            output_root, "timings_" + str(int(data_size)) + ".txt")

        args_string = "--timing_output_path " + output_path
        args_string += " --num_bytes " + str(int(data_size))
        run_string = binary_path + ' ' + args_string
        print("Running experiment as:\n " + run_string)
        subprocess.call(run_string, shell=True)


if __name__ == '__main__':
    run()
