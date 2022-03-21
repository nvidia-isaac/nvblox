#!/usr/bin/python3

from genericpath import exists
import os
import wget
from zipfile import ZipFile
import argparse

import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d

from nvblox.datasets.threedmatch.loader import ThreeDMatchLoader
from nvblox.conversions import open3d_conversions
import nvblox

from icecream import ic


def get_colored_cubes_mesh(values, XYZ, cube_size):
    # Normalizing/Clipping the values
    percentile_lim_upper = 90
    percentile_lim_lower = 10
    max_val = np.percentile(values, percentile_lim_upper)
    min_val = np.percentile(values, percentile_lim_lower)
    values_normalized = (values - min_val) / (max_val - min_val)
    values_normalized = values_normalized.clip(min=0.0, max=1.0)
    # Create the values mesh
    slice_mesh = o3d.geometry.TriangleMesh()
    for dist, xyz in zip(values_normalized, XYZ):
        box = o3d.geometry.TriangleMesh.create_box(width=cube_size,
                                                   height=cube_size,
                                                   depth=cube_size)
        color = plt.cm.viridis(dist)
        box.compute_vertex_normals()
        box.paint_uniform_color(color[0:3])
        box.translate(np.array([xyz[0], xyz[1], xyz[2]]))
        slice_mesh += box
    return slice_mesh


def main(num_frames: int):

    # Check for data (and download it if required)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_root = os.path.join(script_dir, "data")
    dataset_name = 'sun3d-mit_76_studyroom-76-1studyroom2'
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.isdir(dataset_dir):
        url = 'http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip'
        print('Downloading dataset from: ' + url)
        wget.download(url, out=data_root)
        print("Extracting dataset")
        ZipFile(dataset_dir + '.zip').extractall(path=data_root)

    # Paths to the data
    sequence_root_path = os.path.join(dataset_dir, "seq-01")
    camera_intrinsics_path = os.path.join(
        dataset_dir, "camera-intrinsics.txt")
    loader = ThreeDMatchLoader(sequence_root_path, camera_intrinsics_path)

    # Create a tsdf_layer (our map/reconstruction)
    block_size = nvblox.utils.block_size(voxel_size=0.05)
    tsdf_layer = nvblox.TsdfLayer(
        block_size, memory_type=nvblox.MemoryType.Device)
    mesh_layer = nvblox.MeshLayer(
        block_size, memory_type=nvblox.MemoryType.Unified)

    # TSDF Integrator
    integrator = nvblox.ProjectiveTsdfIntegrator()

    # Integrating all the frames
    if num_frames is None:
        num_frames = loader.get_number_of_frames()
    for frame_idx in range(num_frames):
        print(f"Integrating frame: {frame_idx} / {num_frames}")
        (depth_frame, T_L_C_mat, camera) = loader.get_frame_data(frame_idx)
        if np.isnan(T_L_C_mat).any() or np.isnan(depth_frame).any():
            continue
        (depth_frame, T_L_C_mat, camera) = loader.get_frame_data(frame_idx)
        integrator.integrateFrame(depth_frame, T_L_C_mat, camera, tsdf_layer)

    # Mesh the tsdf_layer and safe the result as ply to a file
    print("Extracting mesh from TSDF")
    mesh_integrator = nvblox.mesh.MeshIntegrator()
    mesh_integrator.integrateMeshFromDistanceField(tsdf_layer, mesh_layer)

    # Generate an ESDF
    esdf_layer = nvblox.EsdfLayer(
        tsdf_layer.block_size, memory_type=nvblox.MemoryType.Unified)
    esdf_integrator = nvblox.EsdfIntegrator()
    esdf_integrator.integrate_layer(tsdf_layer, esdf_layer)

    # Get the bounding box of the layer
    aabb = esdf_layer.get_aabb_of_observed_voxels()

    # Get the 3D locations of horizontal slice
    # Note: 3DMatch is Y-axis up, so a vertical slice is along the y axis
    resolution_m = 0.1
    slice_point_ratio = 0.35
    slice_point_m = slice_point_ratio * aabb.sizes()[1] + aabb.min[1]
    X, Z = np.mgrid[aabb.min[0]:aabb.max[0]:resolution_m,
                    aabb.min[2]:aabb.max[2]:resolution_m]
    X, Y, Z = np.broadcast_arrays(X, slice_point_m, Z)
    XYZ = np.stack((X.flatten(), Y.flatten(), Z.flatten())).transpose().astype(np.float32)

    #  Interpolate the ESDF at slice locations
    slice, flags = nvblox.interpolate_on_cpu(XYZ, esdf_layer)

    # Convert the mesh to Open3D and visualize
    mesh = open3d_conversions.mesh_layer_to_open3d(mesh_layer)
    mesh.compute_vertex_normals()
    slice_mesh = get_colored_cubes_mesh(
        slice[flags == True], XYZ[flags == True], cube_size=3.0 * resolution_m / 4.0)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.add_geometry(open3d_conversions.aabb_to_open3d(aabb))
    vis.add_geometry(slice_mesh)
    vis.run()
    vis.destroy_window()

    # Write the mesh to file
    mesh_path = os.path.join(script_dir, "output", dataset_name + ".ply")
    print("Writing the mesh to: " + mesh_path)
    nvblox.io.output_mesh_layer_to_ply(mesh_layer, mesh_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument("--number_of_frames", metavar="number_of_frames", type=int,
                        help="Number of frames to process. Not including this arg mean process all frames.")
    args = parser.parse_args()
    main(args.number_of_frames)
