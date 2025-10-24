# Changelog

All releases of the nvblox library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Unreleased (public branch)

- Option for initializing freespace voxels to free.
- Camera sensor extended with support for distortion (radial and tangential).
- Support externally defined sensor types.
- Dynamic object detection now supports lidar sensor.
- nvblox_torch supported for Jetpack 6.
- Support CUDA13 and Blackwell architectures.
- Color and Feature integrators use exponential filter instead of weighted average.

## [v.0.0.8] - Date: 2025-06-17

- Bugfix in blocks-to-update to prevent occasional segfaults for high-res reconstructions.
- Switch docs framework to Sphinx
- Remove alpha channel from input RGB images
- Fix stdgpu export
- Support for building on Ubuntu 22 and 24
- Use ccache for improved build speed
- Add support for deep feature integration
- Add nvblox torch module
- Add pre-commit for automatic lint checking
- GPU version of ground plane estimator

## [v.0.0.7] - Date: 2024-12-05

### Added

- Add support for masked depth in TSDF integrator.
- CHECK-style macros in device code.
- Add blocks in view caching.
- Add layer streamer class.
- Add handling of any number of GPU hash collissions.
- Add workspace bounds for manipulator usecases.


### Changed

- Modernize the cmake build system.
- Separate index tracker for mesh.
- Masked depth image to reduce size of occupancy layer.
- Batch initialization of recycled blocks.


### Fixed

- Memory-type fixes in image and unif-vector + resizing.
- Fix memory leak test hang due to excessive stream creation.
- Reduce cuda stream creation for dynamics.
- Hashmap resizing fixed.


## [v.0.0.6] - Date: 2024-05-08

### Added

- Add end-to-end benchmark script.
  - Generate KPIs directly from single invocation of script
- Add block pool allocation during startup.
- Add data structure to store 3d ESDF grid.
- Add GPU serialization of voxel layers.
  - Mesh serializer is generalized in order to support serialization of voxel-block layers.
- Add and integrate image cache.
  - Allow image cache to have cached images of different sizes at the same time.
- Add delay measurement feature.


### Changed

- Change decay to have view-based voxel exclusion.
- Change weighting function to linear with max weighting from squared dropoff.
- Move parameters to separate class.
- Update to jetpack 6.


### Fixed

- Fix unified ptr async cloner.
  - 10 percent speedup for replica integration
- Disable multithreading in image reader.
- Disable checks for mapping type in human mapper.
  - Support for multi-cam as we run humans only in one camera.
- Support external CMAKE_CUDA_ARCHITECTURES.
- Fix image buffers allocation for dynamics.


## [v.0.0.5] - Date: 2023-10-18

### Added

- Finished baseline for dynamic detection.
  - Integrate full depth to static mapper
  - Ignoring esdf sites in freespace
- Added a class for effective serialization of a mesh layer.
- Added surrounding radius clearing for the occupancy layer.
  - Move radius clearing to common function upstream
  - Add unit test to check the working
- Add optional preprocessing to the input depth image to dilate the invalid regions.
  - This addresses depth bleeding issues we saw when using the realsense 455 on carter.
- Add TSDF decay integrator
  - Generalized the existing occupancy decayer to also support TSDF decay.
- Add a method for getting the names of all rate tickers.
  - Used in the GXF wrapper to get all timer names to send to sight.
- Add function to decay all occupancy voxels, without any excluded voxels.
- Benchmark GPU<->CPU transfer of mono image.
- checkNppErrors macro.
  - Similar to checkCudaErrors, we use a separate macro for unified handling of npp errors.
- Add dynamics to fuser.
  - Move more functionality into multi mapper (to have a cleaner interface to GXF/ROS/fuser)
  - Add parameter structs and parameter default values.
- Remove small components from mask image.
  - Introduce function for removing small components from mask image.
  - Computation times on Jetson, 640x480 Real mask image:  2ms Worst-case image: 4ms.
- Add useful multi mapper functions.
- Add optional preprocessing to the input depth image to dilate the invalid regions.
  This addresses depth bleeding issues we saw when using the realsense 455 on carter.
- Removed separable compilation of device code in order to support a wider range of toolchains.
- Added test that prevents us from introducing more work on the default CUDA stream.
- Support for executing on a user provided CUDA stream to avoid
  triggering device-wide synchronizations that comes with using the
  default stream. Async versions of copy and memset operations have
  been added to container classes in order to support this.
- Dynamic detection from freespace:
  - DynamicsDetection object which can be used to detect and visualize dynamic objects.
  - FreespaceIntegrator object to update a freespace layer.
  - Updating the DynamicsDetection to rely on the freespace layer for detection.
- CHANGELOG.md
- MeshStreamer object which can be used to limit the bandwidth of the transmitted mesh.


### Changed

- Changed mapper saving functions to not update/alter the map before saving.
  - Add function to mapper to save the TSDF. Used in GXF to service request from sight.
- Change dynamic integration distance to limit computation time.
- Moved and refactored esdf slicing functions to a EsdfSlicer object.
- Removed unnecessary copy functions of Meshblock.
- Refactored ProjectiveIntegrator to simplify the dataflow.
- Turn on shadowing warnings.


### Fixed

- Make changes to get the image masker working if the mask and depth images have different resolutions.
  - Make changes in image masker to point to correct column values during image access
  - Parameterize image masker test to include different test versions
- On Jetson/Orin,the CPU and GPU cannot simultaneously access managed
  memory which causes a segfault in the placement-new operator.
- Warnings stemming from external are suppressed by including the using -isystem.
  - Fixed remaining warnings. Two categories of NVCC warnings are still suppressed.
- Fix missing CUDAToolkit dependency for CUDA::nppc.
- Improved bad-path error-handling in the dataset loaders.
- Variable shadowing in the ProjectiveColorIntegrator.
