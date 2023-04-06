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

import copy
from typing import Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
import matplotlib.cm as cm


def calculate_per_vertex_error(reconstructed_mesh: o3d.geometry.TriangleMesh,
                               groundtruth_mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    """Calculates per-vertex errors between a reconstructed mesh and it's groundtruth.

    Args:
        reconstructed_mesh (o3d.geometry.TriangleMesh): The reconstructed mesh.
        groundtruth_mesh (o3d.geometry.TriangleMesh): The ground-truth mesh.

    Returns:
        np.ndarray: The per-vertex errors
    """
    # Calculating: Error
    # The minimum distance between vertices of the reconstructed mesh
    # and the groundtruth mesh.
    # gt_kdtree = KDTree(gt_pointcloud.points)
    gt_kdtree = KDTree(groundtruth_mesh.vertices)
    errors, _ = gt_kdtree.query(reconstructed_mesh.vertices)
    return errors


def get_error_mesh(reconstructed_mesh: o3d.geometry.TriangleMesh, errors: np.ndarray) -> o3d.geometry.PointCloud:
    """Creates a error mesh, which is a copy of the reconstruction colored by error.

    Args:
        reconstructed_mesh (o3d.geometry.TriangleMesh): Reconstruction as a mesh
        errors (np.ndarray): A list of per vertex errors

    Returns:
        o3d.geometry.PointCloud: The error mesh
    """
    assert(len(reconstructed_mesh.vertices) == len(errors))
    # NOTE(alexmillane): We scale the errors to between the 1st and 99th percentile so the worst outliers don't bias the errors.
    errors_low = np.percentile(errors, 1)
    error_high = np.percentile(errors, 99)
    errors_normalized = (errors - errors_low) / (error_high - errors_low)
    errors_normalized = np.clip(errors_normalized, 0.0, 1.0)
    error_colors = cm.hot(errors_normalized)
    error_mesh = copy.deepcopy(reconstructed_mesh)
    error_mesh.vertex_colors = o3d.utility.Vector3dVector(error_colors[:, 0:3])
    return error_mesh


def get_coverage_mesh(mesh: o3d.geometry.TriangleMesh, within_threadhold_distance_flags: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Returns a mesh colored by which parts are "covered" by reconstruction

    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh to color
        within_threadhold_distance_flags (np.ndarray): Boolean array of flags 
            indicating if vertex is covered

    Returns:
        o3d.geometry.TriangleMesh: Mesh colored by coverage
    """
    assert(len(mesh.vertices) == len(within_threadhold_distance_flags))
    coverage_colors = cm.cool(within_threadhold_distance_flags.astype(float))
    coverage_mesh = copy.deepcopy(mesh)
    coverage_mesh.vertex_colors = o3d.utility.Vector3dVector(
        coverage_colors[:, 0:3])
    coverage_mesh.compute_vertex_normals()
    return coverage_mesh


def get_per_vertex_coverage(reconstructed_mesh: o3d.geometry.TriangleMesh,
                            gt_mesh: o3d.geometry.TriangleMesh, 
                            covered_threshold_m: float = 0.05) -> Tuple[float, np.array]:
    """Gets the proportion of groundtruth mesh vertices "covered" by the reconstructed mesh.
       We a define a covered vertex in the groundtruth mesh as having a reconstructed mesh
       vertex within some threshold.

    Args:
        reconstructed_mesh (o3d.geometry.TriangleMesh): _description_
        gt_mesh (o3d.geometry.TriangleMesh): _description_
        covered_threshold_m (float, optional): _description_. Defaults to 0.05.

    Returns:
        Tuple[float, np.array]: _description_
    """
    reconstruction_kdtree = KDTree(reconstructed_mesh.vertices)
    distances, _ = reconstruction_kdtree.query(gt_mesh.vertices)
    within_threshold_distance_flags = distances < covered_threshold_m
    num_vertices_within_threshold_distance = np.sum(
        within_threshold_distance_flags.astype(int))
    proportion_of_vertices_within_threshold_distance = num_vertices_within_threshold_distance / \
        len(distances)
    return proportion_of_vertices_within_threshold_distance, within_threshold_distance_flags
