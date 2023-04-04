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


def visualize_pcd(pcd_path: str):
    # Load the mesh.
    ptcloud = o3d.io.read_point_cloud(pcd_path)
    print(ptcloud)

    # Create a window.
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(ptcloud)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualize a PCD mesh.")
    parser.add_argument("path", metavar="path", type=str,
                        help="Path to the .pcd file or file to visualize.")

    args = parser.parse_args()
    if args.path:
        visualize_pcd(args.path)
