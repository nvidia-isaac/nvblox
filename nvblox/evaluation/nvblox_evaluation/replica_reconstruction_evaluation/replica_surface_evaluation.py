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
import argparse

import json
import numpy as np
import open3d as o3d

from nvblox_evaluation.evaluation_utils import quad_mesh
from nvblox_evaluation.evaluation_utils import surface_evaluation
from nvblox_evaluation.replica_reconstruction_evaluation.replica_reconstruction import replica_reconstruction
import nvblox_evaluation.replica_reconstruction_evaluation.replica as replica


def evaluate_mesh(reconstructed_mesh_path: Path,
                  groundtruth_mesh_path: Path,
                  output_root_path: Path = None,
                  do_error_visualization: bool = True,
                  do_coverage_visualization: bool = False,
                  covered_threshold_m: float = 0.05) -> None:
    """Calculates error between a reconstruction and the groundtruth geometry.

    Args:
        reconstructed_mesh_path_str (Path): Path to the reconstructed surface to be evaluated.

        groundtruth_mesh_path (Path): Path to the groundtruth mesh.

        output_root_path (Path, optional): Path to the folder in which to store results.
            Note we create a further subfolder with the datasets name. Defaults to None.

        do_visualization (bool, optional): Whether or not to display the error mesh
            in 3D. Defaults to True.

        do_coverage_visualization (bool, optional): Whether or not to display the coverage
            mesh in 3D. Defaults to False.

        covered_threshold_m (float, optional): Distance at which we consider a vertex in
            the groundtruth mesh covered by a vertex in the reconstructed mesh.
            Defaults to 0.05.
    """

    # Detecting dataset name
    dataset_name = replica.get_dataset_name_from_groundtruth_mesh_path(
        groundtruth_mesh_path)
    print(f"Detected dataset name as: {dataset_name}")

    # Path: Output directory
    output_dir = replica.get_output_dir(dataset_name, output_root_path)

    # Load: the reconstructed mesh
    print(f"Loading the reconstruction at: {reconstructed_mesh_path}")
    reconstructed_mesh = o3d.io.read_triangle_mesh(
        str(reconstructed_mesh_path))

    # Load: the groundtruth mesh
    print(f"Loading the groundtruth mesh at: {groundtruth_mesh_path}")
    gt_mesh = quad_mesh.load_quad_mesh(groundtruth_mesh_path)

    # Calculating: Error
    print('Calculating error')
    per_vertex_errors = surface_evaluation.calculate_per_vertex_error(
        reconstructed_mesh, gt_mesh)

    # Calculating: Coverage
    print('Calculating coverage')
    coverage, coverage_flags = surface_evaluation.get_per_vertex_coverage(
        reconstructed_mesh, gt_mesh, covered_threshold_m)

    # Error Mesh (reconstructed mesh colored by error)
    print(f"Coloring an error mesh")
    error_mesh = surface_evaluation.get_error_mesh(
        reconstructed_mesh, per_vertex_errors)

    # Write out the error mesh
    error_mesh_output_path = output_dir / 'error_mesh.ply'
    print(f"Writing the error mesh to: {error_mesh_output_path}")
    o3d.io.write_triangle_mesh(str(error_mesh_output_path), error_mesh)

    # Statistics of the per_vertex_errors
    statistics_dict = {'surface_error_mean': np.mean(per_vertex_errors),
                       'surface_error_median': np.median(per_vertex_errors),
                       'surface_error_max': np.max(per_vertex_errors),
                       'surface_error_min': np.min(per_vertex_errors),
                       'surface_error_percentile_1': np.percentile(per_vertex_errors, 1),
                       'surface_error_percentile_10': np.percentile(per_vertex_errors, 10),
                       'surface_error_percentile_90': np.percentile(per_vertex_errors, 90),
                       'surface_error_percentile_99': np.percentile(per_vertex_errors, 99),
                       'surface_coverage': coverage
                       }
    print("\nReconstructed vertices to GT vertices: error statistics")
    print("-------------------------------------------------------")
    for name, value in statistics_dict.items():
        print(f"{name:<20}{value:0.4f}")

    # Write the results to a JSON
    output_statistics_path = output_dir / 'surface_error_statistics.json'
    print(f"Writing the error statistics to: {output_statistics_path}")
    with open(output_statistics_path, "w") as statistics_file:
        json.dump(statistics_dict, statistics_file, indent=4)

    # Write raw errors to a file
    errors_path = output_dir / 'surface_errors.txt'
    print(f'Writing errors to: {errors_path}')
    np.savetxt(errors_path, per_vertex_errors)

    # Visualization
    if do_error_visualization:
        o3d.visualization.draw_geometries([error_mesh])
    if do_coverage_visualization:
        coverage_mesh = surface_evaluation.get_coverage_mesh(
            gt_mesh.as_open3d, coverage_flags)
        o3d.visualization.draw_geometries([coverage_mesh])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="""Reconstruct a mesh from the replica dataset and test it 
                       against ground-truth geometry.""")

    parser.add_argument("groundtruth_mesh_path", type=Path,
                        help="Path to the groundtruth mesh.")
    parser.add_argument("reconstructed_mesh_path", type=Path, nargs='?',
                        help="Path to the mesh to evaluate.")
    parser.add_argument("--output_root_path", type=Path,
                        help="Path to the directory in which to save results.")
    parser.add_argument("--dont_visualize_error_mesh", dest="do_error_visualization", action='store_const',
                        const=False, default=True,
                        help="Flag indicating if we should visualize the error mesh.")
    parser.add_argument("--do_coverage_visualization", dest="do_coverage_visualization", action='store_const',
                        const=True, default=False,
                        help="Flag indicating if we should display the coverage mesh.")
    parser.add_argument("--fuse_replica_binary_path", type=Path,
                        help="Path to the fuse_replica binary. If not passed we search the standard build folder location.")
    parser.add_argument("--covered_threshold_m", type=Path, default=0.05,
                        help="Distance at which we consider a vertex in the groundtruth mesh covered by a vertex in the reconstructed mesh. Defaults to 0.05.")

    args = parser.parse_args()

    # If no reconstruction is passed in, build one
    if args.reconstructed_mesh_path is None:
        args.reconstructed_mesh_path, _ = replica_reconstruction(
            replica.get_dataset_root_from_groundtruth_mesh_path(
                args.groundtruth_mesh_path),
            output_root_path=args.output_root_path,
            fuse_replica_binary_path=args.fuse_replica_binary_path)

    evaluate_mesh(args.reconstructed_mesh_path,
                  args.groundtruth_mesh_path,
                  output_root_path=args.output_root_path,
                  do_error_visualization=args.do_error_visualization,
                  do_coverage_visualization=args.do_coverage_visualization)
