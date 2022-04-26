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
#include <nvblox/integrators/projective_tsdf_integrator.h>

#include "nvblox/core/color.h"
#include "nvblox/core/cuda/error_check.cuh"
#include "nvblox/core/interpolation_2d.h"
#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

ProjectiveTsdfIntegrator::ProjectiveTsdfIntegrator()
    : ProjectiveIntegratorBase() {
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

ProjectiveTsdfIntegrator::~ProjectiveTsdfIntegrator() {
  finish();
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void ProjectiveTsdfIntegrator::finish() const {
  cudaStreamSynchronize(integration_stream_);
}

void ProjectiveTsdfIntegrator::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    TsdfLayer* layer, std::vector<Index3D>* updated_blocks) {
  CHECK_NOTNULL(layer);
  timing::Timer tsdf_timer("tsdf/integrate");
  // Metric truncation distance for this layer
  const float voxel_size =
      layer->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  // Identify blocks we can (potentially) see (CPU)
  timing::Timer blocks_in_view_timer("tsdf/integrate/get_blocks_in_view");
  const std::vector<Index3D> block_indices =
      view_calculator_.getBlocksInImageViewRaycast(
          depth_frame, T_L_C, camera, layer->block_size(),
          truncation_distance_m, max_integration_distance_m_);
  blocks_in_view_timer.Stop();

  // Allocate blocks (CPU)
  timing::Timer allocate_blocks_timer("tsdf/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, layer);
  allocate_blocks_timer.Stop();

  // Update identified blocks
  // Calls out to the child-class implementing the integation (CPU or GPU)
  timing::Timer update_blocks_timer("tsdf/integrate/update_blocks");
  updateBlocks(block_indices, depth_frame, T_L_C, camera, truncation_distance_m,
               layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

__device__ inline bool updateVoxel(const float surface_depth_measured,
                                   TsdfVoxel* voxel_ptr,
                                   const float voxel_depth_m,
                                   const float truncation_distance_m,
                                   const float max_weight) {
  // Get the MEASURED depth of the VOXEL
  const float voxel_distance_measured = surface_depth_measured - voxel_depth_m;

  // If we're behind the negative truncation distance, just continue.
  if (voxel_distance_measured < -truncation_distance_m) {
    return false;
  }

  // Read CURRENT voxel values (from global GPU memory)
  const float voxel_distance_current = voxel_ptr->distance;
  const float voxel_weight_current = voxel_ptr->weight;

  // NOTE(alexmillane): We could try to use CUDA math functions to speed up
  // below
  // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE

  // Fuse
  constexpr float measurement_weight = 1.0f;
  float fused_distance = (voxel_distance_measured * measurement_weight +
                          voxel_distance_current * voxel_weight_current) /
                         (measurement_weight + voxel_weight_current);

  // Clip
  if (fused_distance > 0.0f) {
    fused_distance = fmin(truncation_distance_m, fused_distance);
  } else {
    fused_distance = fmax(-truncation_distance_m, fused_distance);
  }
  const float weight =
      fmin(measurement_weight + voxel_weight_current, max_weight);

  // Write NEW voxel values (to global GPU memory)
  voxel_ptr->distance = fused_distance;
  voxel_ptr->weight = weight;
  return true;
}

__global__ void integrateBlocks(const Index3D* block_indices_device_ptr,
                                const Camera camera, const float* image,
                                int rows, int cols, const Transform T_C_L,
                                const float block_size,
                                const float truncation_distance_m,
                                const float max_weight,
                                const float max_integration_distance,
                                TsdfBlock** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  if (!projectThreadVoxel(block_indices_device_ptr, camera, T_C_L, block_size,
                          &u_px, &voxel_depth_m)) {
    return;
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  float image_value;
  if (!interpolation::interpolate2DLinear<float>(image, u_px, rows, cols,
                                                 &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order such
  // that adjacent threads (x-major) access adjacent memory locations in the
  // block (z-major).
  TsdfVoxel* voxel_ptr = &(block_device_ptrs[blockIdx.x]
                               ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type
  updateVoxel(image_value, voxel_ptr, voxel_depth_m, truncation_distance_m,
              max_weight);
}

void ProjectiveTsdfIntegrator::updateBlocks(
    const std::vector<Index3D>& block_indices, const DepthImage& depth_frame,
    const Transform& T_L_C, const Camera& camera,
    const float truncation_distance_m, TsdfLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);

  if (block_indices.empty()) {
    return;
  }
  const int num_blocks = block_indices.size();

  // Expand the buffers when needed
  if (num_blocks > block_indices_device_.size()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    block_indices_device_.reserve(new_size);
    block_ptrs_device_.reserve(new_size);
    block_indices_host_.reserve(new_size);
    block_ptrs_host_.reserve(new_size);
  }

  // Stage on the host pinned memory
  block_indices_host_ = block_indices;
  block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, layer_ptr);

  // Transfer to the device
  block_indices_device_ = block_indices_host_;
  block_ptrs_device_ = block_ptrs_host_;

  // We need the inverse transform in the kernel
  const Transform T_C_L = T_L_C.inverse();

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;
  // clang-format off
  integrateBlocks<<<num_thread_blocks, kThreadsPerBlock, 0, integration_stream_>>>(
      block_indices_device_.data(),
      camera,
      depth_frame.dataConstPtr(),
      depth_frame.rows(),
      depth_frame.cols(),
      T_C_L,
      layer_ptr->block_size(),
      truncation_distance_m,
      max_weight_,
      max_integration_distance_m_,
      block_ptrs_device_.data());
  // clang-format on
  checkCudaErrors(cudaPeekAtLastError());

  // Finish processing of the frame before returning control
  finish();
}

}  // namespace nvblox
