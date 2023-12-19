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

import numpy as np
import open3d as o3d
from moviepy.editor import ImageSequenceClip

from nvblox_common.voxel_grid import VoxelGrid


def get_z_slice_animation_clip(
        voxel_grid: VoxelGrid,
        mesh: o3d.geometry.TriangleMesh = None,
        viewpoint: o3d.camera.PinholeCameraParameters = None
) -> ImageSequenceClip:
    """Creates a image sequence containing horizontal slices moving through the z dimension of the VoxelGrid

    Args:
        mesh (o3d.geometry.TriangleMesh, optional): Additional mesh to add to the animation. Defaults to None.
        viewpoint (o3d.camera.PinholeCameraParameters, optional): Viewpoint to record the slice from. Defaults to None.

    Returns:
        ImageSequenceClip: sequence of images of the slicing results
    """
    images = []
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    if viewpoint is not None:
        ctr = vis.get_view_control()
        ctr.convert_from_pinhole_camera_parameters(viewpoint)
    if mesh is not None:
        vis.add_geometry(mesh)
    slice_mesh = o3d.geometry.TriangleMesh()
    vis.add_geometry(slice_mesh)
    images = []
    first = True
    for z_idx in range(voxel_grid.shape()[2]):
        vis.remove_geometry(slice_mesh, reset_bounding_box=False)
        slice_mesh = voxel_grid.get_slice_mesh_at_index(z_idx,
                                                        axis='z',
                                                        cube_size=1.0)
        if first and mesh is None:
            vis.add_geometry(slice_mesh, reset_bounding_box=True)
            first = False
        else:
            vis.add_geometry(slice_mesh, reset_bounding_box=False)
        if viewpoint is not None:
            ctr.convert_from_pinhole_camera_parameters(viewpoint)
        vis.poll_events()
        vis.update_renderer()
        image_float = np.asarray(vis.capture_screen_float_buffer())
        image_uint8 = (image_float * 255).astype(np.uint8)
        images.append(image_uint8)
    vis.destroy_window()
    return ImageSequenceClip(images, fps=10)
