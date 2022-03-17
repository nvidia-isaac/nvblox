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

#include "nvblox/core/camera.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/image.h"
#include "nvblox/core/types.h"

#include "nvblox/experiments/integrators/cuda/depth_frame_texture.cuh"

namespace nvblox {
namespace experiments {

class IntegratorInputFrameExperimentsBase {
 public:
  IntegratorInputFrameExperimentsBase(
      const std::vector<Index3D>& block_indices,
      const DepthImage& depth_frame, const Transform& T_L_C,
      const Camera& camera, const float truncation_distance_m,
      const float max_weight, TsdfLayer* layer_ptr, cudaStream_t stream);
  virtual ~IntegratorInputFrameExperimentsBase();

  // Params for this call
  const float block_size;
  const float truncation_distance_m;
  const float max_weight;
  const int num_blocks;

  // Device-side inputs
  Index3D* block_indices_device_ptr;
  Camera* camera_device_ptr;
  Eigen::Matrix3f* R_C_L_device_ptr;
  Eigen::Vector3f* t_C_L_device_ptr;
  VoxelBlock<TsdfVoxel>** block_device_ptrs;

  // Host-side inputs
  const Eigen::Matrix3f R_C_L;
  const Eigen::Vector3f t_C_L;
  std::vector<VoxelBlock<TsdfVoxel>*> block_ptrs;

  // The stream on which host->device transfers are processed
  cudaStream_t transfer_stream;
};

class IntegratorInputFrameExperimentsTexture
    : public IntegratorInputFrameExperimentsBase {
 public:
  IntegratorInputFrameExperimentsTexture(
      const std::vector<Index3D>& block_indices,
      const DepthImage& depth_frame, const Transform& T_L_C,
      const Camera& camera, const float truncation_distance_m,
      const float max_weight, TsdfLayer* layer_ptr, cudaStream_t stream);
  virtual ~IntegratorInputFrameExperimentsTexture() {};

  // Texture
  DepthImageTexture depth_texture;
};

class IntegratorInputFrameExperimentsGlobal
    : public IntegratorInputFrameExperimentsBase {
 public:
  IntegratorInputFrameExperimentsGlobal(
      const std::vector<Index3D>& block_indices,
      const DepthImage& depth_frame, const Transform& T_L_C,
      const Camera& camera, const float truncation_distance_m,
      const float max_weight, TsdfLayer* layer_ptr, cudaStream_t stream);
  virtual ~IntegratorInputFrameExperimentsGlobal() {};

  // Depth frame
  // Stored as a unified ptr to row-major float memory.
  const float* depth_frame_unified_ptr;
  const int depth_frame_rows;
  const int depth_frame_cols;
};

}  // namespace experiments
}  // namespace nvblox