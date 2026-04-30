# Release Notes

## [v.0.0.10] - Date: 2026-04-16

This release introduces a native, low-latency visualizer module to nvblox.

**nvblox_renderer** is a Vulkan-based visualization library for nvblox data. It renders depth images, color images, point clouds, and triangle meshes using shared CUDA-Vulkan memory, avoiding unnecessary copies between the GPU compute and graphics pipelines.

## [v.0.0.9] - Date: 2026-01-27

This release brings a major overhaul of sensor handling to **nvblox**.

### Sensor upgrades

- LiDAR is now a first-class sensor in nvblox, delivering full feature parity with depth cameras.
- This upgrade unlocks **dynamic object detection**, enabling LiDAR-equipped robots to operate in dynamic, real-world environments.
- To further elevate LiDAR performance, we introduce **motion compensation**, which unwraps LiDAR point clouds using the robot’s motion.
- Camera support has been enhanced with the addition of **radial and tangential distortion parameters**, extending the range of vision platforms supported by nvblox.
- Additionally, nvblox now supports **externally defined sensor types**, making it possible to integrate proprietary or specialized sensors into your perception stack.

### Additional improvements

- nvblox now builds and runs on the latest NVIDIA DGPUs, including the Blackwell generation.
- nvblox_torch is now supported on Jetpack 6.
- Refactored and optimized blocks-to-update tracker for improved performance.
- Option for initializing freespace voxels to free.
- Color and Feature integrators now use exponential filter instead of weighted average.



## [v.0.0.8] - Date: 2025-06-17

This release introduces a major new feature that makes **nvblox** significantly easier to use: a
fast, zero-copy **Python/PyTorch interface**. It is available via **pip install**, which greatly
simplifies installation and setup.

To further open the door for integration with AI workflows, we have also added support for **deep
feature integration** into the nvblox map. This enables the use of general-purpose spatial feature
extractors such as [RADIO-AM](https://github.com/NVlabs/RADIO).

![](docs/images/desk_radio_x2_600px.gif)

*Deep Feature-based Reconstruction using RADIO-AM*


## [v.0.0.7] - Date: 2024-12-05

In this release, several runtime optimizations increased the efficiency of nvblox.
These optimizations are leveraged in [Isaac Perceptor](https://developer.nvidia.com/isaac/perceptor)
and [Isaac Manipulator](https://developer.nvidia.com/isaac/manipulator)
which rely on the 3D reconstruction provided by nvblox.

In the following sections we first list nvblox runtime optimization and then showcase how they are applied to Isaac Perceptor and Isaac Manipulator.

### Nvblox Optimizations

Runtime optimizations increase the efficiency of nvblox and allow it to operate on **low-power**,
**multi-camera** systems with **higher voxel resolution**.

Optimization points:
- Viewpoint cache to reduce redundant ray-cast operations.
- Pre-allocating of voxels in a memory pool to avoid costly on-the-fly memory allocations.
- Bandwidth-limited voxel streaming to allow transmission of voxel maps to remote machines.
- Bulk initialization of voxels to eliminate significant launch overhead when new areas are added to the map.
- Device compaction added to GPU->CPU voxel streaming pipeline to eliminate fragmented global memory reads.
- Parallel kernel launches to maximize utilization of available GPU resources.
- Eliminating all default-stream synchronization points and reducing the amount of local-stream synchronizations.
  A new pre-merge unit will now ensure no unintended synchronizations are introduced.
- Ensure that all calls to third-party CUDA libraries are made asynchronously.
  This required an upgrade of the STDGPU library which forms the backend of the voxel map.
- Add user-provided workspace bounds to reduce unnecessary compute.
- Support ESDF/visualization on request for reduced latency and compute.

For a more details, please refer to the [CHANGELOG](CHANGELOG.md).

### Isaac Perceptor

Dynamic object detection allows nvblox to handle moving obstacles that would otherwise corrupt the cost map.
**Dynamic detection is now enabled by default in Perceptor.**

![](docs/images/dynamic_office_reconstruction.gif)

*Reconstructing an office environment. Dynamic voxels are highlighted in red.*

Support for **multi-RealSense reconstruction** (up to 4 cameras on `Jetson AGX Orin`).

![](docs/images/multi_realsense_galileo.gif)

*Reconstruction integrating data from 4 RealSense cameras simultaneously.*

**Support for `Jetson Nano`** brings nvblox to a lower price point,
which will enable 3D perception for consumer-grade products.

![](docs/images/jetson_nano_reconstruction.gif)

*Live reconstruction running on Nano with visualization data streamed to Foxglove over WiFi.*


### Isaac Manipulator

Nvblox enables collision avoidance in manipulation use-cases:

![](docs/images/nvblox_manipulation.gif)

Performance improvements allow **reconstruction at 1 cm voxel resolution** in workspaces up to `8 m^3` and depth integration at `30 Hz` on `Jetson AGX Orin`.
Support of multi-camera RealSense integration reduces occlusions and increase reconstruction fidelity.


Adding support for pick and place by enabling contact with objects by selective exclusion from the collision field:

![](docs/images/collision_field_exclusion.gif)
