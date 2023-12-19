#!/usr/bin/python3

import argparse
from pathlib import Path

from nvblox_common.voxel_grid import VoxelGrid
from nvblox_common.visualizations import visualize_esdf_voxel_grid, visualize_occupancy_voxel_grid

parser = argparse.ArgumentParser(
    description="Visualize a VoxelGrid stored as ply or npz file.")
parser.add_argument("file_path",
                    type=Path,
                    help="Path to the file to visualize.")
parser.add_argument("--max_visualization_dist_vox",
                    type=int,
                    default=2,
                    help="Max. distance in voxels to the surface "
                    "to show a voxel in the esdf pointcloud.")
parser.add_argument(
    "--visualize_as_occupancy",
    action='store_const',
    const=True,
    default=False,
    help="Whether to interpret and visualize the voxel grid as occupancy grid. "
    "This flag is ignored when loading a npz file (as the is_occupancy_grid flag is indicating the type)."
)

args = parser.parse_args()

# Load the file
file_extension = args.file_path.suffix.lstrip(".")
if file_extension == "npz":
    print("Loading npz file:", args.file_path)
    voxel_grid = VoxelGrid.createFromNpz(args.file_path)
elif file_extension == "ply":
    print("Loading ply file:", args.file_path)
    voxel_grid = VoxelGrid.createFromPly(args.file_path)
    voxel_grid.is_occupancy_grid = args.visualize_as_occupancy
else:
    raise Exception("Unknown file extension: " + file_extension)

# Visualize the voxel grid
if voxel_grid.is_occupancy_grid:
    print("Visualizing the voxel grid as occupancy point cloud.")
    visualize_occupancy_voxel_grid(voxel_grid)
else:
    print("Visualizing the voxel grid as esdf point cloud.")
    visualize_esdf_voxel_grid(voxel_grid, args.max_visualization_dist_vox)
