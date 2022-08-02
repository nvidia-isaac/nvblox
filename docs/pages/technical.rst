# Input/Outputs

Here we discuss the inputs you have to provide to nvblox, and the outputs it produces for downstream tasks.

Inputs:
* **Depth Images**: We require input from a sensor supplying depth per pixel. Examples of such sensors are the Intel Realsense series and Kinect cameras.
* **Sensor Pose**: We require localization of the depth sensor as input to nvblox
* [Optional] **Color Images**: These can be used to color the reconstruction for visualization.

Outputs:
* **Distance map slice**: A 2D map that expresses the distance at each point from objects reconstructed in the environment. This is typically used by a planner to compute a collision cost map.
* **Mesh**: We output a mesh for visualization in RVIZ.

The figure below shows a simple system utilizing nvblox for path planning.

<div align="center"><img src="docs/images/system_diagram.png" width=800px/></div>




# (Brief) Technical Details

Nvblox builds the reconstructed map in the form of a Truncated Signed Distance Function (TSDF) stored in a 3D voxel grid. This approach is similar to 3D occupancy grid mapping approaches in which occupancy probabilities are stored at each voxel. In contrast however, TSDF-based approaches (like nvblox) store the (signed) distance to the closest surface at each voxel. The surface of the environment can then be extracted as the zero-level set of this voxelized function. Typically TSDF-based reconstructions provide higher quality surface reconstructions.

In addition to their use in reconstruction, distance fields are also useful for path planning because they provide an immediate means of checking whether potential future robot positions are in collision. This fact, the utility of distance functions for both reconstruction and planning, motivates their use in nvblox (a reconstruction library for path planning).
