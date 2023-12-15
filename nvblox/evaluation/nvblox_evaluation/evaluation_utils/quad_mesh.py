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
import open3d as o3d
import trimesh

def load_quad_mesh(mesh_filepath: Path) -> o3d.geometry.TriangleMesh:
    """Loads a mesh with quads

    Args:
        mesh_filepath (Path): path to the mesh

    Returns:
        o3d.geometry.TriangleMesh: Triangulized mesh in Open3D format.
    """
    # NOTE(alexmillane): We have to go through trimesh because Open3D can't open quad meshes
    gt_mesh_trimesh = trimesh.load_mesh(str(mesh_filepath))
    gt_mesh = gt_mesh_trimesh.as_open3d

    # Need to copy the verties since the trimesh array is read-only
    gt_mesh.vertex_normals = o3d.utility.Vector3dVector(
        gt_mesh_trimesh.vertex_normals.copy())
    return gt_mesh
