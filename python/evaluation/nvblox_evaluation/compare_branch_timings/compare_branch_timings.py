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
import subprocess
from datetime import datetime

import git

import helpers.run_threedmatch as threedmatch
from helpers.plot_timing_comparison import plot_timings


def generate_timings(dataset_path: str, other_branch_or_hash: str,
                     num_runs: int) -> None:

    # Get this repo
    # Note(alexmillane): We assume that this script is being executed within the repo under test.
    this_directory = os.getcwd()
    repo = git.Repo(this_directory, search_parent_directories=True)
    git_root_dir = repo.git.rev_parse("--show-toplevel")
    assert not repo.bare

    # Branches to time
    current_branch_name = repo.active_branch.name
    branch_names = [current_branch_name, other_branch_or_hash]

    # Where to place timings
    datetime_str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    output_root = os.path.join(this_directory, 'output', datetime_str)

    # Timing each branch
    build_dir = os.path.join(git_root_dir, 'nvblox/build')
    for branch_name in branch_names:

        # Checkout
        print('Checking out: ' + branch_name)
        try:
            repo.git.checkout(branch_name)
        except git.GitCommandError:
            print("Could not checkout branch: " + branch_name +
                  ". Maybe you have uncommited changes?.")
            return

        # Build
        if not os.path.exists(build_dir):
            print("Please create a build space at: " + build_dir)
            return
        build_command = f"cd {build_dir} && cmake .. && make -j16"
        subprocess.call(build_command, shell=True)

        # Benchmark
        branch_str = branch_name.replace('/', '_')
        output_dir = os.path.join(output_root, branch_str)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        threedmatch_binary_path = os.path.join(build_dir,
                                               'executables/fuse_3dmatch')
        threedmatch.run_multiple(num_runs,
                                 threedmatch_binary_path,
                                 dataset_path,
                                 output_dir,
                                 warmup_run=True)

    # Reset to the original branch
    print('Checking out: ' + current_branch_name)
    try:
        repo.git.checkout(current_branch_name)
    except git.GitCommandError:
        pass

    # Plot this stuff
    plot_timings(output_root)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description=
        "Generate a graph which compares the timers between the current and another branch/hash."
    )
    parser.add_argument("dataset_base_path",
                        metavar="dataset_base_path",
                        type=str,
                        help="Path to the 3DMatch dataset root directory.")
    parser.add_argument(
        "other_branch_or_hash",
        metavar="other_branch_or_hash",
        type=str,
        help="The branch name or commit hash to compare against.")
    parser.add_argument(
        "--num_runs",
        metavar="num_runs",
        type=int,
        default=5,
        help="The number of experiments over which to average the timings.")

    args = parser.parse_args()
    generate_timings(args.dataset_base_path, args.other_branch_or_hash,
                     args.num_runs)
