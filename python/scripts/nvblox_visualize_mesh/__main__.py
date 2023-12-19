#!/usr/bin/python3

import argparse
from pathlib import Path

from nvblox_common.visualizations import visualize_mesh_ply

parser = argparse.ArgumentParser(description="Visualize a PLY mesh.")
parser.add_argument("ply_path",
                    type=Path,
                    help="Path to the ply file to visualize.")

parser.add_argument(
    "--normal_coloring",
    dest="do_normal_coloring",
    action='store_const',
    const=True,
    default=False,
    help="Flag indicating if we should just color the mesh by normals.")

args = parser.parse_args()
visualize_mesh_ply(args.ply_path, args.do_normal_coloring)
