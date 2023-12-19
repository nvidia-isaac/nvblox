#
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

import argparse
from pathlib import Path
import open3d as o3d

from nvblox_common.voxel_grid import VoxelGrid
from nvblox_common.visualizations import visualize_mesh_ply, visualize_esdf_voxel_grid, visualize_occupancy_voxel_grid

parser = argparse.ArgumentParser(
    description="""Convert a mesh to a occupancy or esdf voxel grid.""")
parser.add_argument("mesh_path",
                    type=Path,
                    help="Path to the mesh to convert.")
parser.add_argument(
    "--voxel_grid_npz_storage_path",
    type=Path,
    help="Where to store the esdf as a numpy array (npz file).")
parser.add_argument("--voxel_grid_ply_storage_path",
                    type=Path,
                    help="Where to store the esdf as a ply pointcloud.")
parser.add_argument("--visualize_mesh",
                    action='store_const',
                    const=True,
                    default=False,
                    help="Whether to show the input mesh in open3d.")
parser.add_argument("--visualize_voxel_grid",
                    action='store_const',
                    const=True,
                    default=False,
                    help="Whether to show the output esdf in open3d.")
parser.add_argument(
    "--store_as_occupancy",
    action='store_const',
    const=True,
    default=False,
    help="Whether to store the voxel grid as occupancy instead of esdf grid.")
parser.add_argument("--voxel_size",
                    type=float,
                    default=0.05,
                    help="Voxel size of the resulting grid.")
parser.add_argument("--max_visualization_dist_vox",
                    type=int,
                    default=2,
                    help="Max. distance in voxels to the surface "
                    "to show a voxel in the esdf pointcloud.")
parser.add_argument("--use_bunny_mesh",
                    action='store_const',
                    const=True,
                    default=False,
                    help="Whether to use the bunny mesh to run the script "
                    "(will ignore mesh_path argument).")
args = parser.parse_args()

# Check if we want to run with the bunny mesh
if args.use_bunny_mesh:
    print("Configuring to run script on BunnyMesh")
    args.mesh_path = o3d.data.BunnyMesh().path
    args.voxel_size = 0.0025
    print("Voxel size changed to:", args.voxel_size)

print("Compute the voxel grid from the mesh ply at:", args.mesh_path)
voxel_grid = VoxelGrid.createFromMeshPly(args.mesh_path, args.voxel_size)

if args.store_as_occupancy:
    print("Converting voxel grid to occupancy grid.")
    voxel_grid.convert_voxel_values_to_occupancy()

if args.voxel_grid_npz_storage_path:
    print("Save voxel grid to npz file at:", args.voxel_grid_npz_storage_path)
    voxel_grid.writeToNpz(args.voxel_grid_npz_storage_path)

if args.voxel_grid_ply_storage_path:
    print("Save voxel grid to ply file at:", args.voxel_grid_ply_storage_path)
    voxel_grid.writeToPly(args.voxel_grid_ply_storage_path)

if args.visualize_mesh:
    print("Visualize the mesh")
    visualize_mesh_ply(args.mesh_path)

if args.visualize_voxel_grid:
    if args.store_as_occupancy:
        print("Visualize the converted occupancy voxel grid")
        visualize_occupancy_voxel_grid(voxel_grid)
    else:
        print("Visualize the converted esdf voxel grid")
        visualize_esdf_voxel_grid(voxel_grid, args.max_visualization_dist_vox)
