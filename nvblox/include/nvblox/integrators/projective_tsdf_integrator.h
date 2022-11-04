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

#include "nvblox/core/blox.h"
#include "nvblox/core/camera.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/image.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/lidar.h"
#include "nvblox/core/oslidar.h"
#include "nvblox/core/types.h"
#include "nvblox/core/voxels.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/integrators/projective_integrator_base.h"
#include "nvblox/integrators/view_calculator.h"

namespace nvblox {

/// A class performing TSDF intregration
///
/// Integrates a depth images into TSDF layers. The "projective" is a describes
/// one type of integration. Namely that voxels in view are projected into the
/// depth image (the alternative being casting rays out from the camera).
class ProjectiveTsdfIntegrator : public ProjectiveIntegratorBase {
 public:
  ProjectiveTsdfIntegrator();
  virtual ~ProjectiveTsdfIntegrator();

  /// Integrates a depth image in to the passed TSDF layer.
  /// @param depth_frame A depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param camera A the camera (intrinsics) model.
  /// @param layer A pointer to the layer into which this observation will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Camera& camera, TsdfLayer* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Integrates a depth image in to the passed TSDF layer.
  /// @param depth_frame A depth image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param lidar A the LiDAR model.
  /// @param layer A pointer to the layer into which this observation will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(const DepthImage& depth_frame, const Transform& T_L_C,
                      const Lidar& lidar, TsdfLayer* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Integrates a depth image in to the passed TSDF layer.
  /// @param depth_frame A depth image.
  /// @param height_frame A z image.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param lidar A the Ouster LiDAR model.
  /// @param layer A pointer to the layer into which this observation will be
  /// intergrated.
  /// @param updated_blocks Optional pointer to a vector which will contain the
  /// 3D indices of blocks affected by the integration.
  void integrateFrame(DepthImage& depth_frame, DepthImage& height_frame,
                      const Transform& T_L_C, OSLidar& oslidar,
                      TsdfLayer* layer,
                      std::vector<Index3D>* updated_blocks = nullptr);

  /// Blocks until GPU operations are complete
  /// Ensure outstanding operations are finished (relevant for integrators
  /// launching asynchronous work)
  void finish() const override;

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

 protected:
  // Params
  // NOTE(alexmillane): See the getters above for a description.
  float lidar_linear_interpolation_max_allowable_difference_vox_ = 2.0f;
  float lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ = 0.5f;

  // TODO(jjiao): the main function to implement the GPU-based integration
  // Given a set of blocks in view (block_indices) perform TSDF updates on all
  // voxels within these blocks on the GPU.
  void integrateBlocks(const DepthImage& depth_frame, const Transform& T_C_L,
                       const Camera& camera, TsdfLayer* layer_ptr);
  void integrateBlocks(const DepthImage& depth_frame, const Transform& T_C_L,
                       const Lidar& lidar, TsdfLayer* layer_ptr);
  void integrateBlocks(const DepthImage& depth_frame, const Transform& T_C_L,
                       const OSLidar& lidar, TsdfLayer* layer_ptr);

  template <typename SensorType>
  void integrateBlocksTemplate(const std::vector<Index3D>& block_indices,
                               const DepthImage& depth_frame,
                               const Transform& T_L_C, const SensorType& sensor,
                               TsdfLayer* layer);

  // The internal, templated version of the integrateFrame methods above. Called
  // internally with either a camera or lidar sensor.
  template <typename SensorType>
  void integrateFrameTemplate(const DepthImage& depth_frame,
                              const Transform& T_L_C, const SensorType& sensor,
                              TsdfLayer* layer,
                              std::vector<Index3D>* updated_blocks = nullptr);

  // Blocks to integrate on the current call (indices and pointers)
  // NOTE(alexmillane): We have one pinned host and one device vector and
  // transfer between them.
  device_vector<Index3D> block_indices_device_;
  device_vector<TsdfBlock*> block_ptrs_device_;
  host_vector<Index3D> block_indices_host_;
  host_vector<TsdfBlock*> block_ptrs_host_;

  // CUDA stream to process ingration on
  cudaStream_t integration_stream_;

  // TODO(jjiao): store depth image and z image into GPU memory
  float* depth_frame_ptr_cuda_;
  float* height_frame_ptr_cuda_;
};

}  // namespace nvblox
