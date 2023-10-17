# Changelog

All releases of the nvblox library will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
