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
import sys

import torch
import open3d as o3d
import matplotlib

from nvblox_torch.examples.utils.scenes import get_single_sphere_scene_mapper
from nvblox_torch.examples.utils.interrupt_handling import run_with_graceful_interrupt
from nvblox_torch.indexing import get_voxel_center_grids
from nvblox_torch.visualization import get_voxel_mesh
from nvblox_torch.layer import TsdfLayer, convert_layer_to_dense_tensor

SPHERE_RADIUS_M = 1.0


def convert_to_colors(tensor: torch.Tensor) -> torch.Tensor:
    # Normalize
    min_value = torch.min(tensor)
    max_value = torch.max(tensor)
    normalized_value = (tensor - min_value) / (max_value - min_value)
    # Convert to colors
    cmap = matplotlib.cm.get_cmap('viridis')
    colors = cmap(normalized_value.cpu().numpy())[:, :3]
    colors = torch.from_numpy(colors)
    return colors


def visualize_voxels_sparse_access(tsdf_layer: TsdfLayer, mesh_o3d: o3d.geometry.TriangleMesh,
                                   visualize: bool) -> None:
    # Extract all the voxel blocks as pytorch tensors
    blocks, indices = tsdf_layer.get_all_blocks()

    # Loop over all the voxel blocks
    voxels_centers_meeting_condition = []
    voxel_centers_list = get_voxel_center_grids(indices, tsdf_layer.voxel_size(), device='cuda')
    for block, voxel_centers in zip(blocks, voxel_centers_list):
        # Get the TSDF values
        tsdf_values = block[..., 0]
        # Get the mask of the voxels that are inside the sphere
        mask = tsdf_values < 0.0
        # Append the voxel centers that are inside the sphere
        voxels_centers_meeting_condition.append(voxel_centers[mask, :])
    voxel_center_grid_meeting_condition = torch.cat(voxels_centers_meeting_condition, dim=0)
    print(f'Found {voxel_center_grid_meeting_condition.shape[0]} voxels '
          'inside the sphere using sparse access.')

    # Color by x-coordinate
    colors = convert_to_colors(voxel_center_grid_meeting_condition[:, 0])

    # Convert to an Open3D mesh
    voxels_mesh_o3d = get_voxel_mesh(
        centers=voxel_center_grid_meeting_condition,
        voxel_size_m=tsdf_layer.voxel_size(),
        colors=colors,
    )

    # We translate the voxels to the right to avoid overlapping with the mesh
    voxels_mesh_o3d.translate(torch.tensor([2 * SPHERE_RADIUS_M, 0.0, 0.0]))

    # Visualize the voxels (and the mesh)
    if visualize:
        print('Close the visualize window to continue...')
        o3d.visualization.draw_geometries([mesh_o3d, voxels_mesh_o3d])


def visualize_voxels_dense_access(tsdf_layer: TsdfLayer, mesh_o3d: o3d.geometry.TriangleMesh,
                                  visualize: bool) -> None:
    # Convert the TSDF layer to a dense tensor
    tsdf_dense, voxel_center_grid = convert_layer_to_dense_tensor(layer=tsdf_layer)
    # Get the positions of the voxels that are inside the sphere
    dense_mask = tsdf_dense < 0.0
    voxel_center_grid_meeting_condition = voxel_center_grid[torch.squeeze(dense_mask), :]
    voxel_center_grid_meeting_condition = voxel_center_grid_meeting_condition.reshape(-1, 3)
    print(f'Found {voxel_center_grid_meeting_condition.shape[0]} voxels '
          'inside the sphere using dense access.')
    # Color by y-coordinate
    colors = convert_to_colors(voxel_center_grid_meeting_condition[:, 1])
    # Visualize
    voxels_mesh_o3d = get_voxel_mesh(
        centers=voxel_center_grid_meeting_condition,
        voxel_size_m=tsdf_layer.voxel_size(),
        colors=colors,
    )
    voxels_mesh_o3d.translate(torch.tensor([2 * SPHERE_RADIUS_M, 0.0, 0.0]))
    if visualize:
        print('Close the visualize window to continue...')
        o3d.visualization.draw_geometries([mesh_o3d, voxels_mesh_o3d])


def main(visualize: bool) -> int:
    # Get a dummy scene
    mapper = get_single_sphere_scene_mapper(radius_m=SPHERE_RADIUS_M)
    mapper.update_color_mesh()
    mesh = mapper.get_color_mesh()
    mesh_o3d = mesh.to_open3d()
    mesh_o3d.compute_vertex_normals()

    # Get the TSDF layer
    tsdf_layer = mapper.tsdf_layer_view()

    # Visualize the voxels using dense access
    print('Visualizing voxels using dense access...')
    visualize_voxels_dense_access(tsdf_layer, mesh_o3d, visualize)

    # Visualize the voxels using sparse access
    print('Visualizing voxels using sparse access...')
    visualize_voxels_sparse_access(tsdf_layer, mesh_o3d, visualize)

    print('Done.')

    return 0


if __name__ == '__main__':
    sys.exit(run_with_graceful_interrupt(main, visualize=True))
