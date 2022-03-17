#!/usr/bin/python3

import os
import time
import argparse
import subprocess
from datetime import datetime

import git

import nvblox.experiments.threedmatch as threedmatch

from plot_timings import plot_timings

profiles = {"min": {"tsdf_frame_subsampling": 3, "color_frame_subsampling": 30,
                    "esdf_frame_subsampling": 10, "mesh_frame_subsampling": 30},
            "goal": {"tsdf_frame_subsampling": 1, "color_frame_subsampling": 10,
                     "esdf_frame_subsampling": 3, "mesh_frame_subsampling": 10}}
resolutions = [0.05, 0.10, 0.20]


def gather_timings(dataset_path: str) -> None:
    # Params
    num_runs = 1

    # Get the repo base & all needed paths.
    this_directory = os.getcwd()
    repo = git.Repo(this_directory, search_parent_directories=True)
    git_root_dir = repo.git.rev_parse("--show-toplevel")
    assert not repo.bare
    build_dir = os.path.join(git_root_dir, 'nvblox/build')
    threedmatch_binary_path = os.path.join(
        build_dir, 'experiments/fuse_3dmatch')

    # Where to place timings
    datetime_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    output_root = os.path.join(
        this_directory, 'output', datetime_str)

    # Time each profile
    for profile_name, profile_val in profiles.items():
        folder_name = profile_name

        # Time each resolution within each profile
        # TODO
        # Benchmark
        output_dir = os.path.join(output_root, folder_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        threedmatch.run_multiple(
            num_runs, threedmatch_binary_path, dataset_path, output_dir,
            warmup_run=False, flags=profile_val)

    # Plot this stuff
    plot_timings(output_root)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Generate timing results comparing the current and another branch/hash.")
    parser.add_argument(
        "dataset_base_path", metavar="dataset_base_path", type=str,
        help="Path to the 3DMatch dataset root directory.")

    args = parser.parse_args()
    gather_timings(args.dataset_base_path)
