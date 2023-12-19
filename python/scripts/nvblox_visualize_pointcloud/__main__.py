#!/usr/bin/python3

import argparse
import open3d as o3d
from pathlib import Path

parser = argparse.ArgumentParser(description="Visualize a PLY pointcloud.")
parser.add_argument("ply_path",
                    type=Path,
                    help="Path to the ply file to visualize.")

args = parser.parse_args()

ply_point_cloud = o3d.io.read_point_cloud(str(args.ply_path))
o3d.visualization.draw_geometries([ply_point_cloud])
