#!/usr/bin/python3

import numpy as np
import sys
import open3d as o3d


def main(argv):

    pointcloud_np = np.genfromtxt(argv[1])

    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pointcloud_np)

    # Create a window.
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pointcloud)
    vis.run()


if __name__ == "__main__":
    main(sys.argv)
