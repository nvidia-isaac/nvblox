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


from pathlib import Path
import os
from typing import List
import subprocess
import json
import pandas as pd
import argparse


def run_experiment(dataset_path: Path, voxel_sizes: List[float]) -> None:

    # Find the binary to do the fusion
    script_dir = Path(__file__).resolve().parent
    build_root_dir = script_dir.parents[2] / 'build'
    fuse_3dmatch_binary_path = build_root_dir / 'executables' / 'fuse_3dmatch'

    # Dataset path
    dataset_name = dataset_path.name
    print(f"dataset_name: {dataset_name}")

    results_dict = {}
    output_root = script_dir / 'output' / dataset_name
    for voxel_size_m in voxel_sizes_m:

        voxel_size_mm = voxel_size_m * 1000

        # Output path
        output_dir = output_root / f"{voxel_size_mm}"
        os.makedirs(output_dir, exist_ok=True)

        # Run the fusion
        reconstructed_mesh_path = output_dir / 'mesh.ply'
        reconstructed_map_path = output_dir / 'map.nvblx'
        print(f"Running executable at:\t{fuse_3dmatch_binary_path}")
        print(f"On the dataset at:\t{dataset_path}")
        print(f"Outputting mesh at:\t{reconstructed_mesh_path}")
        print(f"Outputting map file at:\t{reconstructed_map_path}")
        mesh_output_path_flag = "--mesh_output_path"
        map_output_path_flag = "--map_output_path"
        voxel_size_flag = "--voxel_size"
        subprocess.run([f"{fuse_3dmatch_binary_path}", f"{dataset_path}",
                        mesh_output_path_flag, f"{reconstructed_mesh_path}",
                        map_output_path_flag, f"{reconstructed_map_path}",
                        voxel_size_flag, f"{voxel_size_m}"])

        # Run the analysis
        freespace_binary_path = build_root_dir / 'experiments' / \
            'experiments' / 'ratio_of_freespace' / 'ratio_of_freespace'
        print(f"Running executable at:\t{freespace_binary_path}")
        freespace_result = subprocess.run(
            [f"{freespace_binary_path}", f"{reconstructed_map_path}"], capture_output=True)

        # Put the results in a dictionary
        acceptable_keys = [ "voxel size",
                            "num_blocks",
                            "num_freespace",
                            "num_near_surface",
                            "num_partially_observed_freespace",
                            "freespace ratio",
                            "(partially observed) freespace ratio"]
        voxel_size_results_dict = {}
        for line in freespace_result.stdout.decode("utf-8").split('\n'):
            label_result_pair = line.split(': ')
            if (len(label_result_pair)) == 2 and label_result_pair[0] in acceptable_keys:
                voxel_size_results_dict[label_result_pair[0]] = float(
                    label_result_pair[1])

        # Write results out
        json_output_path = output_dir / 'results.json'
        with open(json_output_path, "w") as statistics_file:
            json.dump(voxel_size_results_dict, statistics_file, indent=4)

        # Save results per voxel size
        results_dict[voxel_size_m] = voxel_size_results_dict

    # Converting results-per-voxel-size to a dataframe
    results_list = []
    for _, results in results_dict.items():
        results_list.append(results)
    results_dataframe = pd.DataFrame(results_list)
    results_dataframe.set_index('voxel size')
    print(results_dataframe)

    # Writing out to a human readable table.
    output_path = output_root / 'results.txt'
    results_dataframe.to_markdown(output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Evaluates a reconstructed ESDF.")
    parser.add_argument("dataset_path", type=Path,
                        help="Path to the 3dmatch dataset.")

    args = parser.parse_args()

    print(f"Running experiment on 3D Match dataset at: {args.dataset_path}")

    voxel_sizes_m = [0.1, 0.05, 0.04, 0.03, 0.02, 0.01]
    print(f"Running experiments for voxel sizes: {voxel_sizes_m}")

    run_experiment(args.dataset_path, voxel_sizes_m)
