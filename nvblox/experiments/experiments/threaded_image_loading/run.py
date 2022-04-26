import os
import subprocess
import argparse
from enum import Enum

from plot_comparison import plot_timings


class LoadType(Enum):
    kSingle = 1
    kMulti = 2


def run_single(dataset_base_path: str, timing_output_path: str, load_type: LoadType, num_threads: int = -1, num_images: int = -1):
    # Launch the 3DMatch reconstruction
    experiment_executable_name = 'threaded_image_loading'
    if load_type == LoadType.kSingle:
        memory_flag = '--single_thread'
    else:
        memory_flag = '--multi_thread'
    args_string = dataset_base_path
    args_string += ' ' + memory_flag
    args_string += ' --timing_output_path ' + timing_output_path
    if load_type == LoadType.kMulti:
        args_string += ' --num_threads ' + str(num_threads)
    print(f"num_images: {num_images}")
    if num_images > 0:
        args_string += ' --num_images ' + str(num_images)
    print("args_string: " + args_string)
    subprocess.call(['./' + experiment_executable_name +
                    ' ' + args_string], shell=True)


def run_experiment(dataset_base_path: str, number_of_images: int):

    print(f"number_of_images: {number_of_images}")

    # Create the timings output directory
    this_directory = os.getcwd()
    output_dir = os.path.join(this_directory, 'output')
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    output_path = os.path.join(output_dir, "single_threaded_timings.txt")
    run_single(dataset_base_path, output_path,
               LoadType.kSingle, num_images=number_of_images)

    thread_nums = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for num_threads in thread_nums:
        output_path = os.path.join(
            output_dir, f"multi_threaded_{num_threads:02}_timings.txt")
        run_single(dataset_base_path, output_path,
                   LoadType.kMulti, num_threads=num_threads, num_images=number_of_images)

    plot_timings(output_dir)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument("dataset_base_path", metavar="dataset_base_path", type=str,
                        help="Path to the 3DMatch dataset root directory.")
    parser.add_argument("--number_of_images", metavar="number_of_images", type=int, default=200,
                        help="Number of images to load.")
    args = parser.parse_args()
    if args.dataset_base_path:
        run_experiment(args.dataset_base_path,
                       args.number_of_images)
