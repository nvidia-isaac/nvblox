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
#include <nvblox/integrators/internal/decayer.h>

#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/interpolation/interpolation_2d.h"

namespace nvblox {

namespace {

/// Gets from a layer, the indices of blocks for decay.
///
/// This is either all the blocks, or all the block minus the blocks fitting the
/// block_exclusion_options.
/// @tparam LayerType
/// @param layer_ptr The layer to be decayed
/// @param maybe_block_exclusion_options Specifies which blocks will be excluded
/// from decay.
/// @return A list of the blocks to decay.
template <class LayerType>
std::vector<Index3D> getBlockIndicesToDecay(
    LayerType* layer_ptr, const std::optional<DecayBlockExclusionOptions>&
                              maybe_block_exclusion_options) {
  // If we want to exclude some blocks, do so, otherwise return all blocks in
  // the layer.
  if (maybe_block_exclusion_options) {
    const auto block_exclusion_options = maybe_block_exclusion_options.value();
    // Create a set so we can do fast lookup of blocks to exclude
    Index3DSet excluded_indices_set;
    excluded_indices_set.insert(
        block_exclusion_options.block_indices_to_exclude.begin(),
        block_exclusion_options.block_indices_to_exclude.end());

    const float exclusion_radius_m_sq =
        (block_exclusion_options.exclusion_radius_m &&
         block_exclusion_options.exclusion_center)
            ? (*block_exclusion_options.exclusion_radius_m *
               *block_exclusion_options.exclusion_radius_m)
            : -1.0f;

    // Predicate that returns true for blocks we wish to decay
    auto predicate = [&layer_ptr, &excluded_indices_set,
                      &block_exclusion_options,
                      &exclusion_radius_m_sq](const Index3D& index) {
      if (excluded_indices_set.count(index) == 1) {
        return false;
      }
      if (exclusion_radius_m_sq > 0.0f &&
          block_exclusion_options.exclusion_center) {
        const Vector3f& block_center =
            getPositionFromBlockIndex(layer_ptr->block_size(), index);
        const float dist_sq =
            (block_center - *block_exclusion_options.exclusion_center)
                .squaredNorm();
        return (dist_sq > exclusion_radius_m_sq);
      }
      return true;
    };
    // Return all indices to decay
    return layer_ptr->getBlockIndicesIf(predicate);
  } else {
    return layer_ptr->getAllBlockIndices();
  }
}

/// Returns true if a voxel is in view of the camera, is not occluded, is not
/// out of max range, and has a valid depth measurment.
__device__ bool doesVoxelHaveDepthMeasurement(
    const Index3D* block_indices_device_ptr, const Camera camera,
    const float* image, int rows, int cols, const Transform T_C_L,
    const float block_size, const float max_integration_distance,
    const float truncation_distance_m) {
  // Project the voxel into the depth image
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel(block_indices_device_ptr, camera, T_C_L, block_size,
                          max_integration_distance, &u_px, &voxel_depth_m,
                          &p_voxel_center_C)) {
    return false;
  }
  // Interpolate on the image plane
  // Note that the value of the depth image is the depth to the surface.
  float surface_depth_measured;
  if (!interpolation::interpolate2DClosest<
          float, interpolation::checkers::FloatPixelGreaterThanZero>(
          image, u_px, rows, cols, &surface_depth_measured)) {
    return false;
  }
  // Check the distance from the surface
  const float voxel_to_surface_distance =
      surface_depth_measured - voxel_depth_m;
  // Check that we're not occuluded (we're occluded if we're more than the
  // truncation distance behind a surface).
  if (voxel_to_surface_distance < -truncation_distance_m) {
    return false;
  }
  return true;
}

}  // namespace

template <typename BlockType, typename DecayFunctorType>
__device__ void decay(BlockType** block_ptrs,
                      const DecayFunctorType& voxel_decayer,
                      const bool do_decay, bool* is_block_fully_decayed) {
  // Initialize the output
  __shared__ bool is_block_fully_decayed_shared;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    is_block_fully_decayed_shared = true;
  }
  __syncthreads();

  // Load the voxel from global memory
  typename BlockType::VoxelType* voxel_ptr =
      &(block_ptrs[blockIdx.x]->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);
  assert(voxel_ptr != nullptr);

  // If requested, do the decay step.
  if (do_decay) {
    voxel_decayer(voxel_ptr);
  }

  // If any voxel in the block is not decayed, set the block's decayed
  // status to false. NOTE: There could be more than one thread writing
  // this value, but because all of them write false it is no issue.
  // If voxel *not* fully decayed, indicate block not fully decayed.
  if (!voxel_decayer.isFullyDecayed(voxel_ptr)) {
    is_block_fully_decayed_shared = false;
  }
  __syncthreads();

  // One thread writes the output
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    is_block_fully_decayed[blockIdx.x] = is_block_fully_decayed_shared;
  }
}

template <typename BlockType, typename DecayFunctorType>
__global__ void decayKernel(BlockType** block_ptrs,
                            const DecayFunctorType voxel_decayer,
                            bool* is_block_fully_decayed) {
  constexpr bool kDoDecay = true;
  decay(block_ptrs, voxel_decayer, kDoDecay, is_block_fully_decayed);
}

template <typename BlockType, typename DecayFunctorType>
__global__ void decayTsdfExcludeImageKernel(
    BlockType** block_ptrs,                // NOLINT
    const DecayFunctorType voxel_decayer,  // NOLINT
    const Index3D* block_indices,          // NOLINT
    const Camera camera,                   // NOLINT
    const float* depth_image,              // NOLINT
    const int rows,                        // NOLINT
    const int cols,                        // NOLINT
    const Transform T_C_L,                 // NOLINT
    const float block_size_m,              // NOLINT
    const float max_distance_m,            // NOLINT
    const float truncation_distance_m,     // NOLINT
    bool* is_block_fully_decayed) {
  // We do the decay step, only if the voxel is not in view.
  const bool do_decay = (!doesVoxelHaveDepthMeasurement(
      block_indices, camera, depth_image, rows, cols, T_C_L, block_size_m,
      max_distance_m, truncation_distance_m));
  decay(block_ptrs, voxel_decayer, do_decay, is_block_fully_decayed);
}

template <class LayerType>
template <typename DecayFunctorType>
std::vector<Index3D> VoxelDecayer<LayerType>::decay(
    LayerType* layer_ptr,                         // NOLINT
    const DecayFunctorType& voxel_decay_functor,  // NOLINT
    const bool deallocate_decayed_blocks,         // NOLINT
    const std::optional<DecayBlockExclusionOptions>& block_exclusion_options,
    const std::optional<DecayViewExclusionOptions>& view_exclusion_options,
    const CudaStream cuda_stream) {
  CHECK_NOTNULL(layer_ptr);

  // Get block indices to decay and their block pointers
  const std::vector<Index3D> block_indices_to_decay =
      getBlockIndicesToDecay(layer_ptr, block_exclusion_options);

  const std::vector<typename LayerType::BlockType*> block_ptrs_to_decay =
      getBlockPtrsFromIndices(block_indices_to_decay, layer_ptr);

  if (block_ptrs_to_decay.empty()) {
    // Empty layer, nothing to do here.
    return std::vector<Index3D>();
  }

  expandBuffersIfRequired(
      block_ptrs_to_decay.size(), cuda_stream, &allocated_block_ptrs_host_,
      &allocated_block_ptrs_device_, &block_fully_decayed_device_,
      &block_fully_decayed_host_);

  // Get the block pointers on host and copy them to device
  allocated_block_ptrs_host_.copyFromAsync(block_ptrs_to_decay, cuda_stream);
  allocated_block_ptrs_device_.copyFromAsync(allocated_block_ptrs_host_,
                                             cuda_stream);

  // If we're excluding voxels in view, we also need the block indices on the
  // device (to get their position in space).
  if (view_exclusion_options) {
    CHECK_EQ(block_indices_to_decay.size(), block_ptrs_to_decay.size());
    expandBuffersIfRequired(block_indices_to_decay.size(), cuda_stream,
                            &allocated_block_indices_host_,
                            &allocated_block_indices_device_);
    allocated_block_indices_host_.copyFromAsync(block_indices_to_decay,
                                                cuda_stream);
    allocated_block_indices_device_.copyFromAsync(allocated_block_indices_host_,
                                                  cuda_stream);
  }

  block_fully_decayed_device_.resizeAsync(block_ptrs_to_decay.size(),
                                          cuda_stream);

  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = allocated_block_ptrs_host_.size();
  if (view_exclusion_options) {
    // If the max view and/or truncation distances are not set, we set them to
    // something really high (such that they have no effect).
    const float kernel_max_view_distance_m =
        (view_exclusion_options->max_view_distance_m)
            ? view_exclusion_options->max_view_distance_m.value()
            : std::numeric_limits<float>::max();
    const float kernel_truncation_distance_m =
        (view_exclusion_options->truncation_distance_m)
            ? view_exclusion_options->truncation_distance_m.value()
            : std::numeric_limits<float>::max();
    decayTsdfExcludeImageKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                                  cuda_stream>>>(
        allocated_block_ptrs_device_.data(),                  // NOLINT
        voxel_decay_functor,                                  // NOLINT
        allocated_block_indices_device_.data(),               // NOLINT
        view_exclusion_options->camera,                       // NOLINT
        view_exclusion_options->depth_image->dataConstPtr(),  // NOLINT
        view_exclusion_options->depth_image->rows(),          // NOLINT
        view_exclusion_options->depth_image->cols(),          // NOLINT
        view_exclusion_options->T_L_C.inverse(),              // NOLINT
        layer_ptr->block_size(),                              // NOLINT
        kernel_max_view_distance_m,                           // NOLINT
        kernel_truncation_distance_m,                         // NOLINT
        block_fully_decayed_device_.data()                    // NOLINT
    );
  } else {
    decayKernel<<<num_thread_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
        allocated_block_ptrs_device_.data(),  // NOLINT
        voxel_decay_functor,                  // NOLINT
        block_fully_decayed_device_.data()    // NOLINT
    );
  }
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back to host and synchronize
  block_fully_decayed_host_.copyFromAsync(block_fully_decayed_device_,
                                          cuda_stream);
  cuda_stream.synchronize();

  // Check if nothing is lost on the way
  CHECK(allocated_block_ptrs_host_.size() == block_ptrs_to_decay.size());
  CHECK(allocated_block_ptrs_device_.size() == block_ptrs_to_decay.size());
  CHECK(block_fully_decayed_device_.size() == block_ptrs_to_decay.size());
  CHECK(block_fully_decayed_host_.size() == block_ptrs_to_decay.size());
  if (view_exclusion_options) {
    CHECK(allocated_block_indices_host_.size() == block_ptrs_to_decay.size());
    CHECK(allocated_block_indices_device_.size() == block_ptrs_to_decay.size());
  }

  if (deallocate_decayed_blocks) {
    return deallocateFullyDecayedBlocks(layer_ptr, block_indices_to_decay);
  } else {
    return std::vector<Index3D>();
  }
}

template <class LayerType>
std::vector<Index3D> VoxelDecayer<LayerType>::deallocateFullyDecayedBlocks(
    LayerType* layer_ptr, const std::vector<Index3D>& decayed_block_indices) {
  CHECK(decayed_block_indices.size() == block_fully_decayed_host_.size());

  std::vector<Index3D> deallocated_blocks;
  deallocated_blocks.reserve(decayed_block_indices.size());
  for (size_t i = 0; i < decayed_block_indices.size(); ++i) {
    if (block_fully_decayed_host_[i]) {
      layer_ptr->clearBlock(decayed_block_indices[i]);
      deallocated_blocks.push_back(decayed_block_indices[i]);
    }
  }
  return deallocated_blocks;
}

}  // namespace nvblox
