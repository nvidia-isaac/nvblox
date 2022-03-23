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

import os
import sys
import argparse

import numpy as np
import open3d as o3d


def visualize_ply(ply_path: str, do_normal_coloring: bool):
    # Load the mesh.
    mesh = o3d.io.read_triangle_mesh(ply_path)
    print(mesh)
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize a PLY mesh.")
    parser.add_argument("path", metavar="path", type=str,
                        help="Path to the .ply file or file to visualize.")

    parser.add_argument("--normal_coloring", dest="do_normal_coloring", action='store_const',
                        const=True, default=False,
                        help="Flag indicating if we should just color the mesh by normals.")

    args = parser.parse_args()
    if args.path:
        visualize_ply(args.path, args.do_normal_coloring)
