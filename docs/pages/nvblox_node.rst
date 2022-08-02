
.. # Package Reference
.. In this section we give an overview of parameters, inputs, and outputs within this repository and their purpose and settings.

.. ## nvblox_ros
.. The package centers around the `nvblox_node`, whose parameters, inputs, and outputs are described in detail here.


.. ### Published Topics

.. **mesh** ``nvblox_msgs::msg::Mesh`` \
..   A visualization topic showing the mesh produced from the TSDF in a form that can be seen in RViz using `nvblox_rviz_plugin`. Set ``max_mesh_update_hz`` to control its update rate.

.. **pointcloud** ``sensor_msgs::msg::PointCloud2`` \
..   A pointcloud of the 2D ESDF (Euclidean Signed Distance Field), with intensity as the metric distance to the nearest obstacle. Set ``max_esdf_update_hz`` to control its update rate.

.. **map_slice** ``nvblox_msgs::msg::DistanceMapSlice`` \
..   A 2D slice of the ESDF, to be consumed by `nvblox_nav2` package for interfacing with Nav2. Set ``max_esdf_update_hz`` to control its update rate.

.. ### Subscribed Topics

.. **tf** ``tf2_msgs::msg::TFMessage`` \
..   The pose of the sensor relative to the global frame is resolved through TF2. Please see the [ROS2 documentation](https://docs.ros.org/en/foxy/Tutorials/Tf2/Introduction-To-Tf2.html) for more details.
   
.. **depth/image** ``sensor_msgs::msg::Image`` \
..   The input depth image to be integrated. Must be paired with a `camera_info` message below. Supports both floating-point (depth in meters) and uint16 (depth in millimeters, OpenNI format).

.. **depth/camera_info** ``sensor_msgs::msg::CameraInfo`` \
..   Required topic along with the depth image; contains intrinsic calibration parameters of the depth camera.

.. **color/image** ``sensor_msgs::msg::Image`` \
..   Optional input color image to be integrated. Must be paired with a `camera_info` message below. Only used to color the mesh.

.. **color/camera_info** ``sensor_msgs::msg::CameraInfo`` \
..   Optional topic along with the color image above, contains intrinsics of the color camera.

.. ### Services

.. **save_ply** ``std_srvs::srv::Empty`` \
..   This service has an empty request and response. Calling this service will write a mesh to disk at the path specified by the `output_dir` parameter.

.. ### Parameters

.. A summary of the user settable parameters. All parameters are listed as:

.. **Parameter** `Default` \
..   Description.

.. #### General Parameters

.. **voxel_size** `0.05` \
..   Voxel size (in meters) to use for the map.

.. **esdf** `true` \
..   Whether to compute the ESDF (map of distances to the nearest obstacle).

.. **esdf_2d** `true` \
..   Whether to compute the ESDF in 2D (true) or 3D (false).

.. **distance_slice** `true` \
..   Whether to output a distance slice of the ESDF to be used for path planning. 

.. **mesh** `true` \
..   Whether to output a mesh for visualization in rviz, to be used with `nvblox_rviz_plugin`.

.. **global_frame** `map` \
..   The name of the TF frame to be used as the global frame.

.. **slice_height** `1.0` \
..   The *output* slice height for the distance slice and ESDF pointcloud. Does not need to be within min and max height below. In units of meters.

.. **min_height** `0.0` \
..   The minimum height, in meters, to consider obstacles part of the 2D ESDF slice.

.. **max_height** `1.0` \
..   The maximum height, in meters, to consider obstacles part of the 2D ESDF slice.

.. **max_tsdf_update_hz** `10.0` \
..   The maximum rate (in Hz) at which to integrate depth images into the TSDF. A value of 0.0 means there is no cap.

.. **max_color_update_hz** `5.0` \
..   The maximum rate (in Hz) at which to integrate color images into the color layer. A value of 0.0 means there is no cap.

.. **max_mesh_update_hz** `5.0` \
..   The maximum rate (in Hz) at which to update and color the mesh. A value of 0.0 means there is no cap.

.. **max_esdf_update_hz** `2.0` \
..   The maximum rate (in Hz) at which to update the ESDF and output the distance slice. A value of 0.0 means there is no cap.


.. #### Integrator Settings
.. **tsdf_integrator_max_integration_distance_m** `10.0` \
..   The maximum distance, in meters, to integrate the TSDF up to.

.. **tsdf_integrator_truncation_distance_vox** `4.0` \
..   The truncation distance, in units of voxels, for the TSDF.

.. **tsdf_integrator_max_weight** `100.0` \
..   Maximum weight for the TSDF. Setting this number higher will lead to higher-quality reconstructions but worse performance in dynamic scenes.

.. **mesh_integrator_min_weight** `1e-4` \
..   Minimum weight of the TSDF to consider for inclusion in the mesh.

.. **mesh_integrator_weld_vertices** `false` \
..   Whether to weld identical vertices together in the mesh. Currently reduces the number of vertices by a factor of 5x, but is quite slow so we do not recommend you use this setting.

.. **color_integrator_max_integration_distance_m** `10.0` \
..   Maximum distance, in meters, to integrate the color up to.

.. **esdf_integrator_min_weight** `1e-4` \
..   Minimum weight of the TSDF to consider for inclusion in the ESDF.

.. **esdf_integrator_min_site_distance_vox** `1.0` \
..   Minimum distance to consider a voxel within a surface for the ESDF calculation.

.. **esdf_integrator_max_distance_m** `10.0` \
..   Maximum distance to compute the ESDF up to, in meters.


.. ## nvblox_nav2
.. `nvblox_nav2` consists of two parts: an `nvblox_costmap_layer`, which is a Nav2 costmap plugin, and launch files for running the set-up with Nav2 in the loop (described above).

.. ### `nvblox_costmap_layer` Parameters
.. **nvblox_map_slice_topic** `/nvblox_node/map_slice` \
..   Topic to listen for the slice (set via parameters to allow easy configuration from a parameter file).

.. **max_obstacle_distance** `1.0` \
..   Maximum distance from the surface to consider something to have a collision cost, in meters. This is *NOT* in addition to the inflation distance, but the total.

.. **inflation_distance** `0.5` \
..   Distance to inflate all obstacles by, in meters.

.. # Troubleshooting

.. ## Reconstruction without the planner (troubleshooting)
.. If something isn't working, or as a quick sanity check, you can run reconstruction without the planner.

.. Use the `carter_sim.launch.py` launch file parameters `run\_rviz` and `run\_nav2` to turn on and off running rviz and nav2. To run the reconstruction without the planner (but with rviz), run the following:
.. ```
.. ros2 launch nvblox_nav2 carter_sim.launch.py run_rviz:=true run_nav2:=false
.. ```

.. Now, command the robot to spin in place. In another terminal, source your ROS2 workspace again and enter:
.. ```
.. ros2 topic pub /cmd_vel geometry_msgs/Twist '{linear:  {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0,y: 0.0,z: 0.1}}'
.. ```
.. You should see the robot spin in a circle and reconstruct its environment.


.. ## Do you have a connection to Isaac Sim?
.. A quick way to check if ROS2 is communicating correctly with Isaac Sim is to check whether the depth images are being sent.

.. ```
.. ros2 topic hz /left/depth
.. ```

.. If this does not receive any messages, then something is wrong with ROS2's connection to Isaac Sim. Either the file hasn't been set up correctly (make sure to run `nvblox_isaac_sim/omniverse_scripts/carter_warehouse.py` rather than opening an OV scene manually), or the scene is paused, or there are different ROS2 Domain IDs at play.

