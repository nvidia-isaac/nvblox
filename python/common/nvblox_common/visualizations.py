# Copyright 2023 NVIDIA CORPORATION
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

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nvblox_common.voxel_grid import VoxelGrid


def visualize_mesh_ply(ply_path: Path, do_normal_coloring: bool = False):
    """Visualize a mesh that is stored as ply file in o3d.

    Args:
        ply_path (Path): The path to the ply file.
        do_normal_coloring (bool): Flag indicating if we should just color the mesh by normals.
    """
    # Load the mesh.
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    mesh.compute_vertex_normals()

    # Create a window.
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    # Color the mesh by normals.
    if do_normal_coloring:
        vis.get_render_option().mesh_color_option = \
            o3d.visualization.MeshColorOption.Normal
    vis.run()
    vis.destroy_window()


def visualize_esdf_voxel_grid(voxel_grid: VoxelGrid,
                              max_visualization_dist_vox: int):
    """Visualize a esdf of type VoxelGrid in o3d.

    Args:
        voxel_grid (VoxelGrid): the esdf voxel grid
        max_visualization_dist_vox (int): Max. distance in voxels to the surface 
                                          to show a voxel in the esdf pointcloud.
    """
    assert not voxel_grid.is_occupancy_grid, "This is an occupancy voxel grid. I only visualize esdf voxel grids."
    voxel_centers = voxel_grid.get_valid_voxel_centers()
    intensities = voxel_grid.get_valid_voxel_values()
    voxel_size = voxel_grid.get_voxel_size()

    # Only visualize voxels near surface
    max_visualization_dist_m = voxel_size * max_visualization_dist_vox
    surface_indices = np.where(intensities < max_visualization_dist_m)
    intensities_on_surface = intensities[surface_indices]
    points_on_surface = voxel_centers[surface_indices]

    # Normalize the intensities
    intensities_normalized = (
        intensities_on_surface - intensities_on_surface.min()) / (
            intensities_on_surface.max() - intensities_on_surface.min())

    # Get the colors corresponding to the intensities
    cmap = plt.get_cmap('coolwarm')
    colors = cmap(intensities_normalized)[:, :3]

    # Create a o3d pointcloud and show
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_on_surface)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud])


def visualize_occupancy_voxel_grid(voxel_grid: VoxelGrid):
    """Visualize an occupancy grid of type VoxelGrid in o3d.

    Args:
        voxel_grid (VoxelGrid): the occupancy voxel grid
    """
    assert voxel_grid.is_occupancy_grid, "This is an esdf voxel grid. I only visualize occupancy voxel grids."
    voxel_centers = voxel_grid.get_valid_voxel_centers()
    occupancy_values = voxel_grid.get_valid_voxel_values()

    occupied_voxel_indices = np.where(occupancy_values == True)
    occupied_voxel_centers = voxel_centers[occupied_voxel_indices]

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(occupied_voxel_centers)
    o3d.visualization.draw_geometries([point_cloud])
