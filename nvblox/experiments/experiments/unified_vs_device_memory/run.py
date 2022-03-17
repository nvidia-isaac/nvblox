import os
import subprocess
import argparse
from enum import Enum


class MemoryType(Enum):
    kDevice = 1
    kUnified = 2


def run_single(dataset_base_path: str, timing_output_path: str, memory_type: MemoryType):
    # Launch the 3DMatch reconstruction
    experiment_executable_name = 'unified_vs_device_memory'
    if memory_type == MemoryType.kUnified:
        memory_flag = '--unified_memory'
    else:
        memory_flag = '--device_memory'
    args_string = memory_flag + " --timing_output_path " + \
        timing_output_path + ' ' + dataset_base_path
    print("args_string: " + args_string)
    subprocess.call(['./' + experiment_executable_name +
                    ' ' + args_string], shell=True)


def run_experiment(dataset_base_path: str, number_of_runs: int, warm_up_run: bool):

    # Create the timings output directory
    if not os.path.isdir("output"):
        os.mkdir("output")

    # Run a bunch of times
    run_indices = list(range(number_of_runs))
    if warm_up_run:
        run_indices.insert(0, 0)
    for run_idx in run_indices:
        print("Run: " + str(run_idx) + ", Memory type: kDevice")
        timing_output_path = "./output/device_" + str(run_idx) + ".txt"
        run_single(dataset_base_path, timing_output_path, MemoryType.kDevice)
        print("Run: " + str(run_idx) + ", Memory type: kUnified")
        timing_output_path = "./output/unified_" + str(run_idx) + ".txt"
        run_single(dataset_base_path, timing_output_path, MemoryType.kUnified)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument("dataset_base_path", metavar="dataset_base_path", type=str,
                        help="Path to the 3DMatch dataset root directory.")
    parser.add_argument("--number_of_runs", metavar="number_of_runs", type=int, default=10,
                        help="Number of runs to do in this experiment.")
    args = parser.parse_args()
    warm_up_run = True
    if args.dataset_base_path:
        run_experiment(args.dataset_base_path,
                       args.number_of_runs, warm_up_run)
