# Technical Details

## Input/Outputs

Here we discuss the inputs you have to provide to nvblox, and the outputs it produces for downstream tasks. This is the default setup within ROS 2 for 2D navigation, but note that other outputs are possible (such as the full 3D distance map).

_Inputs_:
* **Depth Images**: (@ref nvblox::Image) We require input from a sensor supplying depth per pixel. Examples of such sensors are the Intel Realsense series and Kinect cameras.
* **Camera Intrinsics**: (@ref nvblox::Camera) The intrinsics associated with the camera that generated the depth (or color) image.
* **Sensor Pose**: (@ref nvblox::Transform) We require localization of the depth sensor as input to nvblox
* [Optional] **Color Images**: (@ref nvblox::Image) These can be used to color the reconstruction for visualization.
* [Optional] **LIDAR**: (@ref nvblox::Lidar) Optionally or instead of Depth images, we can also consume LIDAR pointclouds which are reprojected to a cylindrical depth image based on the LIDAR intrinsics specified in this datatype.

_Outputs_:
* **Distance map slice**: (@ref nvblox::EsdfLayer) A 2D map that expresses the distance at each point from objects reconstructed in the environment. This is typically used by a planner to compute a collision cost map.
* **Mesh**: (@ref nvblox::MeshLayer) We output a mesh for visualization in RVIZ.

The figure below shows a simple system utilizing nvblox for path planning.

![System Diagram](images/system_diagram.png)

## (Brief) Technical Details

Nvblox builds the reconstructed map in the form of a Truncated Signed Distance Function (TSDF) stored in a 3D voxel grid. This approach is similar to 3D occupancy grid mapping approaches in which occupancy probabilities are stored at each voxel. In contrast however, TSDF-based approaches (like nvblox) store the (signed) distance to the closest surface at each voxel. The surface of the environment can then be extracted as the zero-level set of this voxelized function. Typically TSDF-based reconstructions provide higher quality surface reconstructions.

In addition to their use in reconstruction, distance fields are also useful for path planning because they provide an immediate means of checking whether potential future robot positions are in collision. This fact, the utility of distance functions for both reconstruction and planning, motivates their use in nvblox (a reconstruction library for path planning).
