# Changelog

All releases of the nvblox library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased (public branch)

- nvblox_torch supported for Jetpack 6. (#1036)
- Support CUDA13 and Blackwell architectures. (#1042)
- Color and Feature integrators use exponential filter instead of weighted average. (#1025)

## [v.0.0.8] - Date: 2025-06-17

- Bugfix in blocks-to-update to prevent occasional segfaults for high-res reconstructions. (#898)
- Switch docs framework to Sphinx (#864)
- Remove alpha channel from input RGB images (#878)
- Fix stdgpu export (#860)
- Support for building on Ubuntu 22 and 24 (#826, #848)
- Use ccache for imporved build speed (#811)
- Add support for deep feature integration (#793, #854)
- Add nvblox torch module (#691)
- Add pre-commit for automatic lint checking (#766)
- GPU version of ground plane estimator (#763)

## [v.0.0.7] - Date: 2024-12-05

### Added

- Add support for masked depth in TSDF integrator. (#758)
- CHECK-style macros in device code. (#693)
- Add blocks in view caching. (#657)
- Add layer streamer class. (#636)
- Add handling of any number of GPU hash collissions. (#631)
- Add workspace bounds for manipulator usecases. (#600)


### Changed

- Modernize the cmake build system. (#751)
- Separate index tracker for mesh. (#683)
- Masked depth image to reduce size of occupancy layer. (#624)
- Batch initialization of recycled blocks. (#611)


### Fixed

- Memory-type fixes in image and unif-vector + resizing. (#649)
- Fix memory leak test hang due to excessive stream creation. (#618)
- Reduce cuda stream creation for dynamics. (#605)
- Hashmap resizing fixed. (#582)


## [v.0.0.6] - Date: 2024-05-08

### Added

- Add end-to-end benchmark script. (#441)
  - Generate KPIs directly from single invocation of script
- Add block pool allocation during startup. (#468)
- Add data structure to store 3d ESDF grid. (#500)
- Add GPU serialization of voxel layers. (#512)
  - Mesh serializer is generalized in order to support serialization of voxel-block layers.
- Add and integrate image cache. (#526)
  - Allow image cache to have cached images of different sizes at the same time.
- Add delay measurement feature. (#541)


### Changed

- Change decay to have view-based voxel exclusion. (#466)
- Change weighting function to linear with max weighting from squared dropoff. (#476)
- Move parameters to separate class. (#533)
- Update to jetpack 6. (#539)


### Fixed

- Fix unified ptr async cloner. (#456)
  - 10 percent speedup for replica integration
- Disable multithreading in image reader. (#487)
- Disable checks for mapping type in human mapper. (#521)
  - Support for multi-cam as we run humans only in one camera.
- Support external CMAKE_CUDA_ARCHITECTURES. (#549)
- Fix image buffers allocation for dynamics.  (#550)


## [v.0.0.5] - Date: 2023-10-18

### Added

- Finished baseline for dynamic detection. (#367):
  - Integrate full depth to static mapper
  - Ignoring esdf sites in freespace
- Added a class for effective serialization of a mesh layer. (#372)
- Added surrounding radius clearing for the occupancy layer. (#378)
  - Move radius clearing to common function upstream
  - Add unit test to check the working
- Add optional preprocessing to the input depth image to dilate the invalid regions. (#381)
  - This addresses depth bleeding issues we saw when using the realsense 455 on carter.
- Add TSDF decay integrator (#383)
  - Generalized the existing occupancy decayer to also support TSDF decay.
- Add a method for getting the names of all rate tickers. (#387)
  - Used in the GXF wrapper to get all timer names to send to sight.
- Add function to decay all occupancy voxels, without any excluded voxels. (#390)
- Benchmark GPU<->CPU transfer of mono image. (#396)
- checkNppErrors macro. (#398)
  - Similar to checkCudaErrors, we use a separate macro for unified handling of npp errors.
- Add dynamics to fuser. (#399)
  - Move more functionality into multi mapper (to have a cleaner interface to GXF/ROS/fuser)
  - Add parameter structs and parameter default values.
- Remove small components from mask image. (#397)
  - Introduce function for removing small components from mask image.
  - Computation times on Jetson, 640x480 Real mask image:  2ms Worst-case image: 4ms.
- Add useful multi mapper functions. (#404)
- Add optional preprocessing to the input depth image to dilate the invalid regions.
  This addresses depth bleeding issues we saw when using the realsense 455 on carter. (#381)
- Removed separable compilation of device code in order to support a wider range of toolchains. (#379)
- Added test that prevents us from introducing more work on the default CUDA stream. (#347)
- Support for executing on a user provided CUDA stream to avoid
  triggering device-wide synchronizations that comes with using the
  default stream. Async versions of copy and memset operations have
  been added to container classes in order to support this.
- Dynamic detection from freespace:
  - DynamicsDetection object which can be used to detect and visualize dynamic objects. (#336)
  - FreespaceIntegrator object to update a freespace layer. (#355)
  - Updating the DynamicsDetection to rely on the freespace layer for detection. (#361)
- CHANGELOG.md (#338)
- MeshStreamer object which can be used to limit the bandwidth of the transmitted mesh. (#324)


### Changed

- Changed mapper saving functions to not update/alter the map before saving. (#388)
  - Add function to mapper to save the TSDF. Used in GXF to service request from sight.
- Change dynamic integration distance to limit computation time. (#406)
- Moved and refactored esdf slicing functions to a EsdfSlicer object. (#363)
- Removed unnecessary copy functions of Meshblock. (#362)
- Refactored ProjectiveIntegrator to simplify the dataflow. (#354)
- Turn on shadowing warnings. (#358)


### Fixed

- Make changes to get the image masker working if the mask and depth images have different resolutions. (#376)
  - Make changes in image masker to point to correct column values during image access
  - Parameterize image masker test to include different test versions
- On Jetson/Orin,the CPU and GPU cannot simultaneously access managed
  memory which causes a segfault in the placement-new operator. (#386)
- Warnings stemming from external are suppressed by including the using -isystem. (#389)
  - Fixed remaining warnings. Two categories of NVCC warnings are still suppressed.
- Fix missing CUDAToolkit dependency for CUDA::nppc. (#408)
- Improved bad-path error-handling in the dataset loaders. (#353)
- Variable shadowing in the ProjectiveColorIntegrator. (#358)
