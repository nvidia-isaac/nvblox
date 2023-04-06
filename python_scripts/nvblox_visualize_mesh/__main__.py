#!/usr/bin/python3

import argparse
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


parser = argparse.ArgumentParser(description="Visualize a PLY mesh.")
parser.add_argument("ply_path", type=str,
                    help="Path to the ply file to visualize.")

parser.add_argument("--normal_coloring", dest="do_normal_coloring", action='store_const',
                    const=True, default=False,
                    help="Flag indicating if we should just color the mesh by normals.")

args = parser.parse_args()
visualize_ply(args.ply_path, args.do_normal_coloring)
