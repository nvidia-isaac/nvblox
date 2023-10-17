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

import argparse

import subprocess
from pathlib import Path
from typing import Tuple
import json
import pandas as pd

import nvblox_evaluation.replica_reconstruction_evaluation.replica as replica
from nvblox_evaluation.evaluation_utils.parse_nvblox_timing import save_timing_statistics


def replica_reconstruction(dataset_path: Path,
                           output_root_path: Path = None,
                           fuse_replica_binary_path: Path = None,
                           esdf_frame_subsampling : int = 1,
                           mesh_frame_subsampling : int = -1) -> Tuple[Path, Path]:
    """Builds a reconstruction for the replica dataset

    Args:
        dataset_path (Path): Path to the root folder of the dataset.

        output_root_path (Path, optional): Root of the directory where to dump the results.
            Note we create a subfolder below this directory named as the dataset name.
            If no argument is given, an output folder is created below the evaluation
            scripts. Defaults to None.

        fuse_replica_binary_path (Path, optional): The path to the binary which does the
            fusion. Defaults to the build folder. Defaults to None.

        esdf_frame_subsampling (int, optional): How often to compute the ESDF. We compute
            Every N frames. Defaults to 1 (every frame).

        mesh_frame_subsampling (int, optional): How often to compute the Mesh. We compute
            Every N frames. Defaults to -1, which means compute once at the end.

    Raises:
        Exception: If the binary is not found.

    Returns:
        Tuple[Path, Path]: Path to the reconstructed mesh + Path to the reconstructed ESDF. 
    """
    dataset_name = replica.get_dataset_name_from_dataset_root_path(
        dataset_path)

    if fuse_replica_binary_path is None:
        fuse_replica_binary_path = replica.get_default_fuse_replica_binary_path()
    if not fuse_replica_binary_path.is_file():
        raise Exception(f"Cant find binary at:{fuse_replica_binary_path}")

    output_dir = replica.get_output_dir(dataset_name, output_root_path)
    reconstructed_mesh_path = output_dir / 'reconstructed_mesh.ply'
    reconstructed_esdf_path = output_dir / 'reconstructed_esdf.ply'
    timing_path = output_dir / 'timing.txt'

    # Reconstruct the mesh + esdf
    print(f"Running executable at:\t{fuse_replica_binary_path}")
    print(f"On the dataset at:\t{dataset_path}")
    print(f"Outputting mesh at:\t{reconstructed_mesh_path}")
    print(f"Outputting esdf at:\t{reconstructed_esdf_path}")
    mesh_output_path_flag = "--mesh_output_path"
    esdf_output_path_flag = "--esdf_output_path"
    timing_output_path_flag = "--timing_output_path"
    esdf_frame_subsampling_flag = "--esdf_frame_subsampling"
    mesh_frame_subsampling_flag = "--mesh_frame_subsampling"
    subprocess.run([f"{fuse_replica_binary_path}", f"{dataset_path}",
                   mesh_output_path_flag, f"{reconstructed_mesh_path}",
                   esdf_output_path_flag, f"{reconstructed_esdf_path}",
                   timing_output_path_flag, f"{timing_path}",
                   esdf_frame_subsampling_flag, f"{esdf_frame_subsampling}",
                   mesh_frame_subsampling_flag, f"{mesh_frame_subsampling}"])

    # Write timing statistics to JSON
    save_timing_statistics(timing_path, output_dir)

    return reconstructed_mesh_path, reconstructed_esdf_path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="""Reconstruct a mesh from the replica dataset.""")

    parser.add_argument("dataset_path", type=Path,
                        help="Path to the dataset root folder.")
    parser.add_argument("--output_root_path", type=Path,
                        help="Path to the directory in which to save results.")
    parser.add_argument("--fuse_replica_binary_path", type=Path,
                        help="Path to the fuse_replica binary. If not passed we search the standard build folder location.")

    args = parser.parse_args()

    replica_reconstruction(args.dataset_path,
                           output_root_path=args.output_root_path,
                           fuse_replica_binary_path=args.fuse_replica_binary_path)
