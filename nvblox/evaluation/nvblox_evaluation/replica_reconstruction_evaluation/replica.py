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
from pathlib import Path


def get_output_dir(dataset_name: str, output_root_path: Path = None) -> Path:
    """Gets the output sub-directory based on the dataset name, and (optional)
       output root path. Creates this directory if it doesn't already exist.
       If no output root is passed, we create an return output directory
       underneath the evaluation scripts. 

    Args:
        dataset_name (str): Name of the dataset (will be the name of the sub-directory)
        output_root_path (Path, optional): Root folder underwhich to create the output
                          sub-folder. Defaults to None.

    Returns:
        Path: Path to the output directory.
    """

    if output_root_path is None:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / 'output' / dataset_name
        print(
            f"No output root directory passed, saving below script at:\n\t{output_dir}")
    else:
        output_dir = Path(output_root_path) / dataset_name
        print(f"Output to be saved in: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"Output directory doesn't exist. Creating it at {output_dir}")
        os.makedirs(output_dir)
    return output_dir


def get_default_fuse_replica_binary_path() -> Path:
    """Returns the path where the "fuse_replica" binary usually lives.

    Returns:
        Path: Path to the default fuse_replica binary.
    """
    script_dir = Path(__file__).resolve().parent
    return script_dir.parents[2] / 'build' / 'executables' / 'fuse_replica'


def get_dataset_name_from_groundtruth_mesh_path(groundtruth_mesh_path: Path) -> str:
    """Gets the name of the replica dataset from a path to the ground-truth mesh

    Args:
        groundtruth_mesh_path (Path): Path to the groundtruth mesh

    Returns:
        str: Dataset name.
    """
    dataset_name = groundtruth_mesh_path.stem.split('_')[0]
    return dataset_name


def get_dataset_name_from_dataset_root_path(dataset_root_path: Path) -> str:
    """Gets the name of the replica dataset from a path to the dataset root folder

    Args:
        dataset_root_path (Path): Path to the dataset root folder.

    Returns:
        str: Dataset name.
    """
    return dataset_root_path.name


def get_dataset_root_from_groundtruth_mesh_path(groundtruth_mesh_path: Path) -> Path:
    """Gets the path to the dataset root folder from the path to the grountruth mesh. 

    Args:
        groundtruth_mesh_path (Path): Path to the groundtruth mesh.

    Returns:
        Path: Path to the root folder for the dataset.
    """
    return groundtruth_mesh_path.parent / get_dataset_name_from_groundtruth_mesh_path(groundtruth_mesh_path)
