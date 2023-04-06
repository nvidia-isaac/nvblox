/*
Copyright 2022 NVIDIA CORPORATION

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#pragma once

#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/integrators/view_calculator.h"
#include "nvblox/map/common_names.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/lidar.h"

namespace nvblox {

/// A pure-virtual base-class for the projective occupancy and tsdf integrators.
///
/// Integrators deriving from this base class insert (integrate) image and lidar
/// data into a voxel grid. The "projective" in ProjectiveIntegratorBase refers
/// to the fact that the mode of operation is that voxels in the view frustrum
/// are projected into the image in order to associate each voxel with a data
/// pixel.
template <typename VoxelType>
class ProjectiveIntegrator {
 public:
  ProjectiveIntegrator(){};
  virtual ~ProjectiveIntegrator() {}

  template <typename UpdateFunctor>
  void integrateFrame(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Camera& lidar, UpdateFunctor* op,
                      VoxelBlockLayer<VoxelType>* layer,
                      std::vector<Index3D>* updated_blocks);

  template <typename UpdateFunctor>
  void integrateFrame(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Lidar& lidar, UpdateFunctor* op,
                      VoxelBlockLayer<VoxelType>* layer,
                      std::vector<Index3D>* updated_blocks);

  /// A parameter getter
  /// The maximum allowable value for the maximum distance between the linearly
  /// interpolated image value and its four neighbours. Above this value we
  /// consider linear interpolation failed. This is to prevent interpolation
  /// across boundaries in the lidar image, which causing bleeding in the
  /// reconstructed 3D structure.
  /// @returns the maximum allowable distance in voxels
  float lidar_linear_interpolation_max_allowable_difference_vox() const;

  /// A parameter getter
  /// The maximum allowable distance between a reprojected voxel's center and
  /// the ray performing this integration. Above this we consider nearest
  /// nieghbour interpolation to have failed.
  /// @returns the maximum allowable distance in voxels
  float lidar_nearest_interpolation_max_allowable_dist_to_ray_vox() const;

  /// A parameter setter
  /// see lidar_linear_interpolation_max_allowable_difference_vox()
  /// @param the new parameter value
  void lidar_linear_interpolation_max_allowable_difference_vox(float value);

  /// A parameter setter
  /// see lidar_nearest_interpolation_max_allowable_dist_to_ray_vox()
  /// @param the new parameter value
  void lidar_nearest_interpolation_max_allowable_dist_to_ray_vox(float value);

  /// A parameter getter
  /// The truncation distance parameter associated with this integrator. The
  /// truncation distance also determines the max. distance behind a surface up
  /// until which blocks are integrated.
  /// @returns the truncation distance in voxels
  float truncation_distance_vox() const;

  /// Gets the metric truncation distance which is calculated from truncation
  /// distance in voxels and the input voxel size.
  /// @param voxel_size The voxel size of the layer you want the truncation
  /// distance for.
  /// @return The truncation distance
  float get_truncation_distance_m(float voxel_size) const;

  /// A parameter getter
  /// The maximum distance at which voxels are updated. Voxels beyond this
  /// distance from the camera are not affected by integration.
  /// @returns the maximum intragration distance
  float max_integration_distance_m() const;

  /// A parameter setter
  /// See truncation_distance_vox().
  /// @param truncation_distance_vox the truncation distance in voxels.
  void truncation_distance_vox(float truncation_distance_vox);

  /// A parameter setter
  /// See max_integration_distance_m().
  /// @param max_integration_distance_m the maximum intragration distance in
  /// meters.
  void max_integration_distance_m(float max_integration_distance_m);

  /// Returns the object used to calculate the blocks in camera views.
  const ViewCalculator& view_calculator() const;
  /// Returns the object used to calculate the blocks in camera views.
  ViewCalculator& view_calculator();

 protected:
  // Given a set of blocks in view (block_indices) perform TSDF updates on all
  // voxels within these blocks on the GPU.
  template <typename UpdateFunctor>
  void integrateBlocks(const DepthImage& depth_frame, const Transform& T_C_L,
                       const Camera& camera, UpdateFunctor* op,
                       VoxelBlockLayer<VoxelType>* layer_ptr);
  template <typename UpdateFunctor>
  void integrateBlocks(const DepthImage& depth_frame, const Transform& T_C_L,
                       const Lidar& lidar, UpdateFunctor* op,
                       VoxelBlockLayer<VoxelType>* layer_ptr);

  template <typename SensorType, typename UpdateFunctor>
  void integrateBlocksTemplate(const std::vector<Index3D>& block_indices,
                               const DepthImage& depth_frame,
                               const Transform& T_L_C, const SensorType& sensor,
                               UpdateFunctor* op,
                               VoxelBlockLayer<VoxelType>* layer_ptr);

  template <typename SensorType, typename UpdateFunctor>
  void integrateFrameTemplate(const DepthImage& depth_frame,
                              const Transform& T_L_C, const SensorType& sensor,
                              UpdateFunctor* op,
                              VoxelBlockLayer<VoxelType>* layer,
                              std::vector<Index3D>* updated_blocks = nullptr);

  // Get the child integrator name
  virtual std::string getIntegratorName() const = 0;
  bool integrator_name_initialized_ = false;
  std::string integrator_name_;

  // Params
  // NOTE(alexmillane): See the getters above for a description.
  float lidar_linear_interpolation_max_allowable_difference_vox_ = 2.0f;
  float lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ = 0.5f;
  float truncation_distance_vox_ = 4.0f;
  float max_integration_distance_m_ = 7.0f;

  // Frustum calculation.
  mutable ViewCalculator view_calculator_;

  // Blocks to integrate on the current call (indices and pointers)
  // NOTE(alexmillane): We have one pinned host and one device vector and
  // transfer between them.
  device_vector<Index3D> block_indices_device_;
  device_vector<VoxelBlock<VoxelType>*> block_ptrs_device_;
  host_vector<Index3D> block_indices_host_;
  host_vector<VoxelBlock<VoxelType>*> block_ptrs_host_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;
};

}  // namespace nvblox

#include "nvblox/integrators/internal/impl/projective_integrator_impl.h"
