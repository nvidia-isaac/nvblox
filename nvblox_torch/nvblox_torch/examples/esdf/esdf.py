#!/usr/bin/env python
#
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import argparse
import os
import pathlib
import sys
import time

import numpy as np
import matplotlib
import open3d as o3d
import torch

from nvblox_torch.examples.utils.scenes import get_sun3d_scene_mapper
from nvblox_torch.examples.utils.interrupt_handling import run_with_graceful_interrupt
from nvblox_torch.layer import Layer
from nvblox_torch.constants import constants
from nvblox_torch.visualization import to_open3d_voxel_grid
from nvblox_torch.examples.utils.visualization import ViewPointController
from nvblox_torch.mapper import QueryType

MAX_SDF_FOR_VISUALIZATION = 1.0
MIN_SDF_FOR_VISUALIZATION = 0.0
NUM_SLICE_ANIMATIONS = 2
VIEWPOINT_FILE_PATH = pathlib.Path(__file__).parent / 'viewpoint.json'
DEFAULT_NUM_FRAMES = 500


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Reconstruct a feature mesh'
                                     'from the SUN3D dataset.')
    parser.add_argument('--dataset_path',
                        type=pathlib.Path,
                        required=True,
                        help='Path to the dataset/sequence root folder.')
    parser.add_argument('--num_frames',
                        type=int,
                        default=DEFAULT_NUM_FRAMES,
                        help='The number of frames to fuse. '
                        f'If omitted, fuse {DEFAULT_NUM_FRAMES} frames.')
    parser.add_argument('--voxel_size_m',
                        type=float,
                        default=0.05,
                        help='The voxel size in meters.')
    return parser.parse_args()


def get_aabb_voxel_center_grid(layer: Layer) -> torch.Tensor:
    """Gets a grid of points that covers the Axis-Aligned Bounding Box of the passed layer."""
    # Get the limits of the mapped space.
    min_block_idx, max_block_idx = layer.get_block_limits()
    aabb_min_vox = min_block_idx * layer.block_dim_in_voxels
    aabb_max_vox = (max_block_idx + 1) * layer.block_dim_in_voxels
    # Create a 3D grid of points.
    x_linspace = torch.linspace(aabb_min_vox[0],
                                aabb_max_vox[0],
                                aabb_max_vox[0] - aabb_min_vox[0] + 1,
                                dtype=torch.int)
    y_linspace = torch.linspace(aabb_min_vox[1],
                                aabb_max_vox[1],
                                aabb_max_vox[1] - aabb_min_vox[1] + 1,
                                dtype=torch.int)
    z_linspace = torch.linspace(aabb_min_vox[2],
                                aabb_max_vox[2],
                                aabb_max_vox[2] - aabb_min_vox[2] + 1,
                                dtype=torch.int)
    x_grid, y_grid, z_grid = torch.meshgrid(x_linspace, y_linspace, z_linspace, indexing='ij')
    query_grid_xyz_vox = torch.stack([x_grid, y_grid, z_grid], dim=-1)
    # Voxel units to meters.
    # NOTE(alexmillane): We add 0.5 to go from voxel low-side edge to center.
    query_grid_xyz_m = (query_grid_xyz_vox + 0.5) * layer.voxel_size()
    query_grid_xyz_m = query_grid_xyz_m.cuda()
    return query_grid_xyz_m


def to_open3d_esdf_voxel_grid(sdf_values: torch.Tensor, slice_xyz: torch.Tensor,
                              voxel_size_m: float) -> o3d.geometry.VoxelGrid:
    """Converts an ESDF tensor to an open3d voxel grid."""
    # Convert the ESDF to an open3d voxel grid for visualizing.
    cmap = matplotlib.colormaps.get_cmap('plasma')
    sdf_values_normalized = (sdf_values - MIN_SDF_FOR_VISUALIZATION) / (MAX_SDF_FOR_VISUALIZATION -
                                                                        MIN_SDF_FOR_VISUALIZATION)
    sdf_values_normalized = torch.clamp(sdf_values_normalized, MIN_SDF_FOR_VISUALIZATION,
                                        MAX_SDF_FOR_VISUALIZATION)
    slice_colors = cmap(sdf_values_normalized.cpu().numpy())[:, :3] * 255.0
    return to_open3d_voxel_grid(pointcloud=slice_xyz.cpu().numpy(),
                                colors=slice_colors,
                                voxel_size=voxel_size_m)


def set_initial_viewpoint(visualizer: o3d.visualization.Visualizer) -> None:
    """Sets this example's inital viewpoint from file."""
    ctr = visualizer.get_view_control()
    assert os.path.isfile(VIEWPOINT_FILE_PATH), 'Viewpoint file not found'
    param = o3d.io.read_pinhole_camera_parameters(str(VIEWPOINT_FILE_PATH))
    ctr.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)


def main(visualize: bool) -> int:
    """ Main function for the ESDF example query example."""
    args = parse_args()

    # Map a SUN3D scene
    print(f'Mapping {args.dataset_path} with voxel size '
          f'{args.voxel_size_m} and {args.num_frames} frames.')
    mapper = get_sun3d_scene_mapper(
        dataset_path=args.dataset_path,
        voxel_size_m=args.voxel_size_m,
        num_frames=args.num_frames,
    )
    mapper.update_esdf()

    # Get a grid of points that covers the 3D AABB of the mapped space.
    query_grid_xyz_m = get_aabb_voxel_center_grid(mapper.tsdf_layer_view())

    # Query the SDF at each point.
    print('Querying SDF.')
    sdf_values = mapper.query_differentiable_layer(QueryType.ESDF, query_grid_xyz_m.reshape(-1, 3))
    sdf_values = sdf_values.reshape(query_grid_xyz_m.shape[:-1])

    # Get the mask of the points where the query failed.
    valid_mask = torch.logical_not(sdf_values == constants.esdf_unknown_distance())

    # Create the visualization window.
    if visualize:
        view_point_controller = ViewPointController(lookat=np.array([0.0, 0.0, 0.0]))
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        mesh = mapper.get_color_mesh().to_open3d()
        visualizer.add_geometry(mesh)
        set_initial_viewpoint(visualizer)

    # Loop through the slices and visualize the ESDF.
    print(f'Visualizing slices {NUM_SLICE_ANIMATIONS} times.')
    slice_idx_range = np.concatenate(
        [np.arange(sdf_values.shape[2]),
         np.arange(sdf_values.shape[2] - 1, -1, -1)])
    slice_idx_range = np.tile(slice_idx_range, NUM_SLICE_ANIMATIONS)
    for slice_idx in slice_idx_range:

        # Slice the grid.
        slice_mask = valid_mask[..., slice_idx]
        slice_xyz = query_grid_xyz_m[..., slice_idx, :]
        slice_sdf = sdf_values[..., slice_idx]

        # Exclude points that didn't query successfully
        slice_xyz = slice_xyz[slice_mask]
        slice_sdf = slice_sdf[slice_mask]

        num_valid_queries = slice_xyz.shape[0]
        print(f'Slice at index {slice_idx} had {num_valid_queries} valid queries.')

        # Convert the ESDF to an open3d voxel grid for visualizing.
        if num_valid_queries > 0:
            voxel_grid_o3d = to_open3d_esdf_voxel_grid(slice_sdf, slice_xyz, args.voxel_size_m)
            if visualize:
                # Add the geometry and restore the viewpoint.
                view_point_controller.store_camera_pose(visualizer)
                visualizer.clear_geometries()
                visualizer.add_geometry(mesh)
                visualizer.add_geometry(voxel_grid_o3d)
                visualizer.update_renderer()
                view_point_controller.restore_viewpoint(visualizer)
                # Process events and slow things down.
                for _ in range(20):
                    visualizer.poll_events()
                    time.sleep(0.001)

    if visualize:
        visualizer.destroy_window()

    return 0


if __name__ == '__main__':
    sys.exit(run_with_graceful_interrupt(main, visualize=True))
