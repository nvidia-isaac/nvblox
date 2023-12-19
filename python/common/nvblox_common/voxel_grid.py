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
from typing import Tuple

import numpy as np

import open3d as o3d
from plyfile import PlyData, PlyElement
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree as KDTree


# TODO(remos): Move esdf/occupancy specific functionalities to separate subclasses
class VoxelGrid:
    unobserved_sentinal = -1000.0

    def __init__(self,
                 voxels: np.ndarray,
                 min_indices: np.ndarray,
                 voxel_size: float,
                 is_occupancy_grid: bool = False):
        """An object representing an VoxelGrid.

        Args:
            voxels (_type_): A 3D numpy array containing VoxelGrid values
            min_indices (_type_): An 3x1 array representing the low-side corner of the grid in voxel indices.
            voxel_size (_type_): side length of a single voxel.
            is_occupancy_grid (bool, optional): Whether the voxel values represent occupancy or esdf. Defaults to False.
        """
        assert (len(min_indices) == 3)
        self.is_occupancy_grid = is_occupancy_grid
        self.voxels = voxels
        self.min_indices = min_indices
        self.voxel_size = voxel_size

    def shape(self) -> Tuple:
        """Get the grid size.

        Returns:
            _type_: a 3x1 tuple representing the size of the grid
        """
        return self.voxels.shape

    def voxel_centers_along_axis(self, axis_idx: int) -> np.ndarray:
        """Generates a ndarray of the voxel centers along a certain dimension

        Args:
            axis_idx (int): The index of the axis (0, 1, 2)

        Returns:
            np.ndarray: returns a vector of the voxel centers.
        """
        return (
            np.arange(self.min_indices[axis_idx], self.min_indices[axis_idx] +
                      self.shape()[axis_idx]) + 0.5) * self.voxel_size

    def get_valid_voxel_centers(self) -> np.ndarray:
        """Get the centers of all valid voxels as an Nx3 numpy array

        Returns:
            np.ndarray: Nx3 numpy array containing the center locations of valid voxels.
        """
        x_range = self.voxel_centers_along_axis(0)
        y_range = self.voxel_centers_along_axis(1)
        z_range = self.voxel_centers_along_axis(2)
        X, Y, Z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        return np.vstack(
            (X[self.voxels != VoxelGrid.unobserved_sentinal],
             Y[self.voxels != VoxelGrid.unobserved_sentinal],
             Z[self.voxels != VoxelGrid.unobserved_sentinal])).transpose()

    def get_valid_voxel_values(self) -> np.ndarray:
        """Get the centers of all valid voxels as an Nx3 numpy array

        Returns:
            np.ndarray: Nx1 numpy array containing the center locations of valid voxels.
        """
        return self.voxels[self.voxels != VoxelGrid.unobserved_sentinal]

    def get_voxel_size(self) -> float:
        """Get the voxel size of the grid

        Returns:
            float: The voxel size in meters.
        """
        return self.voxel_size

    def convert_voxel_values_to_occupancy(self):
        """Convert the esdf voxel values to occupancy values (true means occupied)
        """
        assert not self.is_occupancy_grid, "Voxel grid is already containing occupancy values. Conversion not valid."
        self.is_occupancy_grid = True
        self.voxels = self.voxels < -0.0

    @staticmethod
    def createFromAABB(aabb: o3d.geometry.AxisAlignedBoundingBox,
                       voxel_size: float, value: float) -> 'VoxelGrid':
        """Generates the smallest VoxelGrid fully enclosing the given AABB with voxels initialized to the passed value.

        Args:
            aabb (o3d.geometry.AxisAlignedBoundingBox): The aabb the VoxelGrid should span.
            voxel_size (float): The voxel size of the VoxelGrid.
            value (float): The value the voxels should be initialized to.

        Returns:
            VoxelGrid: VoxelGrid spanning the AABB initialized to the value.
        """
        max_indices = np.ceil((aabb.get_max_bound() / voxel_size) -
                              0.5).astype(int)
        min_indices = np.floor((aabb.get_min_bound() / voxel_size) -
                               0.5).astype(int)
        dims = max_indices - min_indices
        voxels = np.ones(dims, dtype=int) * value
        return VoxelGrid(voxels, min_indices, voxel_size)

    @staticmethod
    def createFromMeshPly(ply_path: Path, voxel_size: float) -> 'VoxelGrid':
        """Creates a VoxelGrid representing the esdf of a scene stored as mesh ply file.

        Args:
            ply_path (Path): The path to the mesh ply file.
            voxel_size (float): The voxel size of the resulting esdf.

        Returns:
            VoxelGrid: Esdf VoxelGrid resulting from the input mesh.
        """
        mesh = o3d.io.read_triangle_mesh(str(ply_path))
        aabb = mesh.get_axis_aligned_bounding_box()
        # Find voxel centers inside that aabb
        voxel_centers = VoxelGrid.createFromAABB(aabb, voxel_size,
                                                 0).get_valid_voxel_centers()
        # Fill the values with esdf values
        return VoxelGrid.createFromSparseMesh(mesh, voxel_centers)

    @staticmethod
    def createFromSparseMesh(mesh: o3d.geometry.TriangleMesh,
                             points_xyz: np.ndarray) -> 'VoxelGrid':
        """Generates an ESDF VoxelGrid from a triangle mesh at voxel centers passed in. This is
        performed by looking up the closest vertices on the input mesh.

        Args:
            mesh (o3d.geometry.TriangleMesh): Mesh defining the surface.
            points_xyz (np.ndarray): Voxel centers where we want the ESDF calculated.

        Returns:
            VoxelGrid: A voxel grid containing the groundtruth ESDF values.
        """
        # Getting the GT distances using a KD-tree to find the closest points on the mesh surface.
        gt_kdtree = KDTree(mesh.vertices)
        gt_distances, gt_indices = gt_kdtree.query(points_xyz)
        # Getting the signs of the distances
        gt_closest_points = np.asarray(mesh.vertices)[gt_indices]
        gt_closest_vectors = points_xyz - gt_closest_points
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        gt_closest_normals = np.asarray(mesh.vertex_normals)[gt_indices]
        dots = np.sum(np.multiply(gt_closest_vectors, gt_closest_normals),
                      axis=1)
        signs = np.where(dots >= 0, 1.0, -1.0)
        gt_distances = np.multiply(gt_distances, signs)
        return VoxelGrid.createFromSparseVoxels(points_xyz, gt_distances)

    @staticmethod
    def createFromSparseVoxels(voxels_xyz: np.ndarray,
                               voxel_values: np.ndarray) -> 'VoxelGrid':
        """Creates an VoxelGrid object from a list of valid voxels values and their locations.

        Args:
            voxels_xyz (np.ndarray): Nx3 array representing the position of the voxel centers
            voxel_values (np.ndarray): NX1 array representing the value of the voxels.

        Returns:
            VoxelGrid: The object representing the VoxelGrid.
        """
        # Detect the voxel size
        element_wise_diffs = np.diff(voxels_xyz, axis=0).flatten()
        voxel_size = np.min(element_wise_diffs[element_wise_diffs > 0])
        # Convert these two pointcloud parts to our VoxelGrid object
        voxel_indices = (np.around((voxels_xyz / voxel_size) -
                                   0.5)).astype(dtype=np.intc)
        min_indices = np.min(voxel_indices, axis=0)
        max_indices = np.max(voxel_indices, axis=0)
        voxel_indices_zero_based = voxel_indices - min_indices
        sdf = VoxelGrid.unobserved_sentinal * \
            np.ones(max_indices - min_indices + 1)
        sdf[voxel_indices_zero_based[:, 0], voxel_indices_zero_based[:, 1],
            voxel_indices_zero_based[:, 2]] = voxel_values
        return VoxelGrid(sdf, min_indices, voxel_size)

    @staticmethod
    def createFromPly(ply_path: Path) -> 'VoxelGrid':
        """Creates an VoxelGrid object from a nvblox ESDF pointcloud ply.

        Args:
            ply_path (Path): Path to the nvblox file

        Returns:
            VoxelGrid: The object representing the VoxelGrid.
        """
        # Get the xyz position of voxels
        sdf_pointcloud_xyz = np.asarray(
            o3d.io.read_point_cloud(str(ply_path)).points)
        # Get the ESDF values
        sdf_pointcloud_values = np.array(
            PlyData.read(str(ply_path)).elements[0]['intensity'])
        return VoxelGrid.createFromSparseVoxels(sdf_pointcloud_xyz,
                                                sdf_pointcloud_values)

    def writeToPly(self, ply_path: Path) -> None:
        """Writes the ESDF as a pointcloud ply to file.

        Args:
            ply_path (Path): Path to the ply file to write
        """
        xyz = self.get_valid_voxel_centers()
        distances = self.get_valid_voxel_values()
        xyzi = np.hstack([xyz, distances.reshape((-1, 1))])
        xyzi_structured = np.array([tuple(row) for row in xyzi],
                                   dtype=[('x', 'f4'), ('y', 'f4'),
                                          ('z', 'f4'), ('intensity', 'f4')])
        point_elements = PlyElement.describe(xyzi_structured, 'vertex')
        PlyData([point_elements], text=True).write(str(ply_path))

    @staticmethod
    def createFromNpz(npz_path: Path) -> 'VoxelGrid':
        """Creates a VoxelGrid object from a Npz file.

        Args:
            npz_path (Path): Path to the npz file.

        Returns:
            VoxelGrid: The created VoxelGrid
        """
        data = np.load(str(npz_path))
        assert 'voxels' in data.keys(
        ), "Voxels not stored in the loaded Npz file."
        assert 'min_indices' in data.keys(
        ), "Min indices not stored in the loaded Npz file."
        assert 'voxel_size' in data.keys(
        ), "Voxel size not stored in the loaded Npz file."
        assert 'is_occupancy_grid' in data.keys(
        ), "is_occupancy_grid not stored in the loaded Npz file."
        return VoxelGrid(data['voxels'], data['min_indices'],
                         data['voxel_size'], data['is_occupancy_grid'])

    def writeToNpz(self, npz_path: Path):
        """Writes the VoxelGrid as a numpy array to file.

        Args:
            npz_path (Path): Path to the npz file to write
        """
        np.savez_compressed(str(npz_path),
                            voxels=self.voxels,
                            min_indices=self.min_indices,
                            voxel_size=self.voxel_size,
                            is_occupancy_grid=self.is_occupancy_grid)

    def get_slice_mesh_at_ratio(
            self,
            slice_level_ratio: float,
            axis: str = 'x',
            cube_size: float = 0.75) -> o3d.geometry.TriangleMesh:
        """Gets a mesh representing a slice at ratio (0.0-1.0) along dimension axis.

        Args:
            slice_level_ratio (float): Where to slice.
            axis (str, optional): The axis to slice along. Defaults to 'x'.
            cube_size (float, optional): Size of the mesh cube that will be used to
                represent voxels. Given as a fraction of voxel size (i.e between 0.0 and 1.0).

        Returns:
            o3d.geometry.TriangleMesh: Mesh representing the slice.
        """
        assert not self.is_occupancy_grid, "This function should not be called with an occupancy voxel grid."
        assert (slice_level_ratio >= 0.0 and slice_level_ratio <= 1.0)
        assert (axis == 'x' or axis == 'y' or axis == 'z')
        if axis == 'x':
            axis_idx = 0
        elif axis == 'y':
            axis_idx = 1
        else:
            axis_idx = 2
        slice_level_idx = int(self.shape()[axis_idx] * slice_level_ratio)
        return self.get_slice_mesh_at_index(slice_level_idx, axis, cube_size)

    def get_slice_mesh_at_index(
            self,
            slice_level_idx: int,
            axis: str = 'x',
            cube_size: float = 0.75) -> o3d.geometry.TriangleMesh:
        """Gets a mesh representing a slice at slice_level_idx along dimension axis.

        Args:
            slice_level_idx (int): The index to slice at.
            axis (str, optional): The axis to slice along. Defaults to 'x'.
            cube_size (float, optional): Size of the mesh cube that will be used to
                represent voxels. Given as a fraction of voxel size (i.e between 0.0 and 1.0).

        Returns:
            o3d.geometry.TriangleMesh: Mesh representing the slice.
        """
        assert not self.is_occupancy_grid, "This function should not be called with an occupancy voxel grid."
        assert (axis == 'x' or axis == 'y' or axis == 'z')
        assert (cube_size > 0.0 and cube_size <= 1.0)

        # The VoxelGrid values to clip at
        percentile_lim_upper = 90
        percentile_lim_lower = 10
        sdf_clip_max = np.percentile(self.get_valid_voxel_values(),
                                     percentile_lim_upper)
        sdf_clip_min = np.percentile(self.get_valid_voxel_values(),
                                     percentile_lim_lower)

        # Size of the cubes
        voxel_cube_size = self.voxel_size * cube_size

        # Slice
        if axis == 'x':
            slice_level_m = self.voxel_centers_along_axis(0)[slice_level_idx]
            slice = self.voxels[slice_level_idx, :, :]
            dim_1_vec = self.voxel_centers_along_axis(1)
            dim_2_vec = self.voxel_centers_along_axis(2)

            def to_3d(y, z):
                return np.array([slice_level_m, y, z])
        elif axis == 'y':
            slice_level_m = self.voxel_centers_along_axis(1)[slice_level_idx]
            slice = self.voxels[:, slice_level_idx, :]
            dim_1_vec = self.voxel_centers_along_axis(0)
            dim_2_vec = self.voxel_centers_along_axis(2)

            def to_3d(x, z):
                return np.array([x, slice_level_m, z])
        else:
            slice_level_m = self.voxel_centers_along_axis(2)[slice_level_idx]
            slice = self.voxels[:, :, slice_level_idx]
            dim_1_vec = self.voxel_centers_along_axis(0)
            dim_2_vec = self.voxel_centers_along_axis(1)

            def to_3d(x, y):
                return np.array([x, y, slice_level_m])

        # Normalizing/Clipping the distances
        slice_normalized = (slice - sdf_clip_min) / \
            (sdf_clip_max - sdf_clip_min)
        slice_normalized = slice_normalized.clip(min=0.0, max=1.0)

        # Create the slice mesh
        slice_mesh = o3d.geometry.TriangleMesh()
        for idx_1, pos_1 in np.ndenumerate(dim_1_vec):
            for idx_2, pos_2 in np.ndenumerate(dim_2_vec):
                if slice[idx_1, idx_2] == self.unobserved_sentinal:
                    continue
                box = o3d.geometry.TriangleMesh.create_box(
                    width=voxel_cube_size,
                    height=voxel_cube_size,
                    depth=voxel_cube_size)
                color = plt.cm.viridis(slice_normalized[idx_1, idx_2])
                box.compute_vertex_normals()
                box.paint_uniform_color(color[0, 0:3])
                box.translate(to_3d(pos_1, pos_2))
                slice_mesh += box

        return slice_mesh

    def num_valid_voxels(self) -> int:
        """Returns the number of observed voxels

        Returns:
            int: number of observed voxels
        """
        return np.sum(self.voxels != self.unobserved_sentinal)

    def __repr__(self) -> str:
        return "VoxelGrid of voxels with shape: " + str(self.voxels.shape) \
            + " and " + str(self.num_valid_voxels()) + " valid voxels."
