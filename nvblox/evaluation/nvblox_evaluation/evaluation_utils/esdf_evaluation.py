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

import numpy as np
from scipy.spatial import cKDTree as KDTree
import open3d as o3d


from nvblox_evaluation.evaluation_utils.voxel_grid import VoxelGrid


def generate_esdf_from_mesh(gt_mesh: o3d.geometry.TriangleMesh, points_xyz: np.ndarray) -> VoxelGrid:
    """Generates an ESDF from a triangle mesh at voxel centers passed in. This is
       performed by looking up the closest vertices on the input mesh.

    Args:
        gt_mesh (o3d.geometry.TriangleMesh): Mesh defining the surface.
        points_xyz (np.ndarray): Voxel centers where we want the ESDF calculated.

    Returns:
        VoxelGrid: A voxel grid containing the groundtruth ESDF values.
    """
    # Getting the GT distances using a KD-tree to find the closest points on the mesh surface.
    gt_kdtree = KDTree(gt_mesh.vertices)
    gt_distances, gt_indices = gt_kdtree.query(points_xyz)
    # Getting the signs of the distances
    gt_closest_points = np.asarray(gt_mesh.vertices)[gt_indices]
    gt_closest_vectors = points_xyz - gt_closest_points
    gt_closest_normals = np.asarray(gt_mesh.vertex_normals)[gt_indices]
    dots = np.sum(np.multiply(gt_closest_vectors, gt_closest_normals), axis=1)
    signs = np.where(dots >= 0, 1.0, -1.0)
    gt_distances = np.multiply(gt_distances, signs)
    return VoxelGrid.createFromSparseVoxels(points_xyz, gt_distances)


def get_sdf_abs_error_grid(reconstructed_sdf: VoxelGrid, gt_sdf: VoxelGrid) -> VoxelGrid:
    """Get the absolute difference between two grids of distance values. Note
       that we only compute the difference at locations where the ground-truth
       ESDF is positive. (This means that this function is not symmetric in its
       input),

    Args:
        reconstructed_sdf (VoxelGrid): The reconstructed ESDF.
        gt_sdf (VoxelGrid): The ground-truth ESDF.

    Returns:
        VoxelGrid: A voxel grid containing the absolute differences.
    """
    # Check that these two grids define different values for matching voxels.
    # At the moment they even have to be in the same order.
    voxel_center_position_epsilon = 1e-6
    assert(np.max(reconstructed_sdf.get_valid_voxel_centers() -
           gt_sdf.get_valid_voxel_centers()) < voxel_center_position_epsilon)
    # Figure out which voxels we should compare, voxels to compare are:
    # 1) GT positive (dont care if test is negative)
    compare_voxel_flags = gt_sdf.get_valid_voxel_values() >= 0.0
    # Getting the differences
    comparison_xyz = reconstructed_sdf.get_valid_voxel_centers()[
        compare_voxel_flags]
    gt_values = gt_sdf.get_valid_voxel_values()[compare_voxel_flags]
    reconstructed_values = reconstructed_sdf.get_valid_voxel_values()[
        compare_voxel_flags]
    # Getting the differences
    absolute_diffs = np.abs(gt_values - reconstructed_values)
    # Chuck em in a grid
    return VoxelGrid.createFromSparseVoxels(comparison_xyz, absolute_diffs)
