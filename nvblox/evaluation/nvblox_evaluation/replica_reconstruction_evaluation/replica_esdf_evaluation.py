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
from pathlib import Path
import json

import numpy as np
import plotly.express as px
import open3d as o3d

from nvblox_evaluation.evaluation_utils.voxel_grid import VoxelGrid
from nvblox_evaluation.evaluation_utils import esdf_evaluation
from nvblox_evaluation.evaluation_utils import quad_mesh
from nvblox_evaluation.replica_reconstruction_evaluation import replica_reconstruction
from nvblox_evaluation.replica_reconstruction_evaluation import replica


def evaluate_esdf(reconstructed_esdf_path: Path,
                  groundtruth_mesh_path: Path,
                  output_root_path: Path = None,
                  reconstructed_mesh_path: Path = None,
                  do_slice_visualization: bool = True,
                  do_slice_animation: bool = True) -> None:
    """Compares a reconstructed ESDF to a groundtruth ESDF generated from a passed in mesh.

    Args:
        reconstructed_esdf_path (Path): Path to the reconstructed ESDF to evaluate.

        groundtruth_mesh_path (Path): Path to the groundtruth mesh, from which we
            generate a groundtruth ESDF.

        output_root_path (Path, optional): Path where to save the results. Defaults to None.

        reconstructed_mesh_path (Path, optional): Path to a mesh reconstruction. Only used
            for the slice visualization (if requested) Defaults to None.

        do_slice_visualization (bool, optional): Visualize the scene in Open3D. Defaults to True.

        do_slice_animation (bool, optional): Visualize the reconstructed ESDF with an
            animation. Defaults to True.
    """

    # Detecting dataset name
    dataset_name = replica.get_dataset_name_from_groundtruth_mesh_path(
        groundtruth_mesh_path)
    print(f"Detected dataset name as: {dataset_name}")

    # Output path
    if output_root_path is None:
        script_dir = Path(__file__).resolve().parent
        output_dir = script_dir / 'output' / dataset_name
        print(
            f"No output directory passed, saving below script at:\n\t{output_dir}")
    else:
        output_dir = Path(output_root_path) / dataset_name
        print(f"Output to be saved in: {output_dir}")
    if not os.path.exists(output_dir):
        print(f"Output directory doesn't exist. Creating it at {output_dir}")
        os.makedirs(output_dir)

    # Load: Reconstructed the VoxelGrid
    print(f"Loading the reconstructed ESDF at: {reconstructed_esdf_path}")
    reconstructed_sdf = VoxelGrid.createFromPly(reconstructed_esdf_path)

    # Load: Groundtruth mesh
    print(f"Loading the groundtruth mesh at: {groundtruth_mesh_path}")
    gt_mesh = quad_mesh.load_quad_mesh(groundtruth_mesh_path)

    # Get the groundtruth ESDF
    print('Calculating GT ESDF values')
    gt_sdf = esdf_evaluation.generate_esdf_from_mesh(
        gt_mesh, reconstructed_sdf.get_valid_voxel_centers())

    # Get errors
    print('Calculating ESDF errors')
    sdf_abs_diff = esdf_evaluation.get_sdf_abs_error_grid(
        reconstructed_sdf, gt_sdf)
    sdf_abs_diff.writeToPly(output_dir / 'error_esdf.ply')
    abs_errors = sdf_abs_diff.get_valid_voxel_values()

    # Statistics
    statistics_dict = {'esdf_error_mean': np.mean(abs_errors),
                       'esdf_error_median': np.median(abs_errors),
                       'esdf_error_max': np.max(abs_errors),
                       'esdf_error_min': np.min(abs_errors),
                       'esdf_error_percentile_1': np.percentile(abs_errors, 1),
                       'esdf_error_percentile_10': np.percentile(abs_errors, 10),
                       'esdf_error_percentile_90': np.percentile(abs_errors, 90),
                       'esdf_error_percentile_99': np.percentile(abs_errors, 99),
                       'esdf_error_rms': np.sqrt(np.mean(np.square(abs_errors)))
                       }
    print("\ESDF error statistics")
    print("-------------------------------------------------------")
    for name, value in statistics_dict.items():
        print(f"{name:<30}{value:0.4f}")

    # Write the results to a JSON
    output_statistics_path = output_dir / 'esdf_error_statistics.json'
    print(f"Writing the error statistics to: {output_statistics_path}")
    with open(output_statistics_path, "w") as statistics_file:
        json.dump(statistics_dict, statistics_file, indent=4)

    # Error histogram.
    sdf_diff_abs_np = sdf_abs_diff.get_valid_voxel_values()
    np.savetxt(output_dir / 'esdf_errors.txt', sdf_diff_abs_np)
    fig = px.histogram(sdf_diff_abs_np)
    fig.write_image(output_dir / 'esdf_error_histogram.png')
    fig.show()

    # Load: Reconstructed mesh (for use in visualization)
    reconstructed_mesh = None
    if reconstructed_mesh_path is not None:
        reconstructed_mesh = o3d.io.read_triangle_mesh(
            str(reconstructed_mesh_path))
        reconstructed_mesh.compute_vertex_normals()

    # Animation
    if do_slice_animation:
        print('Making a slice animation')
        clip = reconstructed_sdf.get_z_slice_animation_clip(
            mesh=reconstructed_mesh)
        animation_path = output_dir / 'reconstructed_esdf_slice_animation.mp4'
        clip.write_videofile(str(animation_path), fps=10)

    # Slice
    if do_slice_visualization:
        print('Visualizing slice')
        slice_mesh = reconstructed_sdf.get_slice_mesh_at_ratio(
            0.25, axis='z', cube_size=1.0)
        # Create a window.
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        if reconstructed_mesh is not None:
            vis.add_geometry(reconstructed_mesh)
        vis.add_geometry(slice_mesh)
        vis.capture_screen_image(
            str(output_dir / 'reconstructed_esdf_slice.png'), do_render=True)
        vis.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Evaluates a reconstructed ESDF.")
    parser.add_argument("groundtruth_mesh_path", type=Path,
                        help="Path to the groundtruth mesh.")
    parser.add_argument("reconstructed_esdf_path", type=Path, nargs='?',
                        help="Path to the esdf to evaluate.")
    parser.add_argument("reconstructed_mesh_path", nargs='?', default=None, type=Path,
                        help="Path to the reconstructed mesh (for visualization).")
    parser.add_argument("--output_root_path", type=str,
                        help="Path to the directory in which to save results.")
    parser.add_argument("--dont_visualize_slice", dest="do_slice_visualization", action='store_const',
                        const=False, default=True,
                        help="Flag indicating if we should visualize an ESDF slice in 3D.")
    parser.add_argument("--dont_animate_slice", dest="do_slice_animation", action='store_const',
                        const=False, default=True,
                        help="Flag indicating if we should animate an ESDF slice.")
    parser.add_argument("--fuse_replica_binary_path", type=Path,
                        help="Path to the fuse_replica binary. If not passed we search the standard build folder location.")

    args = parser.parse_args()

    # If no reconstruction is passed in, build one
    if args.reconstructed_esdf_path is None:
        args.reconstructed_mesh_path, args.reconstructed_esdf_path = replica_reconstruction(
            replica.get_dataset_root_from_groundtruth_mesh_path(
                args.groundtruth_mesh_path),
            output_root_path=args.output_root_path,
            fuse_replica_binary_path=args.fuse_replica_binary_path)

    evaluate_esdf(args.reconstructed_esdf_path,
                  args.groundtruth_mesh_path,
                  output_root_path=args.output_root_path,
                  reconstructed_mesh_path=args.reconstructed_mesh_path,
                  do_slice_visualization=args.do_slice_visualization,
                  do_slice_animation=args.do_slice_animation)
