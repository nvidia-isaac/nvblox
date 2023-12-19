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

#include <memory>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/log_odds.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// The VoxelDecayer can be used to decay (reduce the weights) of a layer.
/// This is useful for dynamic scenarios, where object might move/disappear
/// when not constantly observed. The class operates by running a per-voxel
/// decay functor over the map.
template <class LayerType>
class VoxelDecayer {
 public:
  VoxelDecayer() = default;
  ~VoxelDecayer() = default;

  /// @brief Does the decay by running the voxel_decay_functor on all voxels.
  /// @tparam DecayFunctorType The (unnamed) type of the functor.
  /// @param layer_ptr The layer to run the decay on.
  /// @param voxel_decay_functor The functor object which does the decay, which
  /// is run on every voxel.
  /// @param deallocate_decayed_blocks If fully decayed blocks should be
  /// deallocated.
  /// @param block_exclusion_options Specifies blocks not to decay.
  /// @param view_exclusion_options Specifies view-based voxel exclusion.
  /// @param cuda_stream The stream to do GPU work on.
  /// @return A vector containing the indices of the blocks deallocated.
  template <typename DecayFunctorType>
  std::vector<Index3D> decay(
      LayerType* layer_ptr,                         // NOLINT
      const DecayFunctorType& voxel_decay_functor,  // NOLINT
      const bool deallocate_decayed_blocks,         // NOLINT
      const std::optional<DecayBlockExclusionOptions>& block_exclusion_options,
      const std::optional<DecayViewExclusionOptions>& view_exclusion_options,
      const CudaStream cuda_stream);

 protected:
  /// Given a vector of blocks that have been decayed, deallocate the ones that
  /// are *fully* decayed (i.e. having a weight that is close to zero)
  /// @param layer_ptr The layer in which to deallocate
  /// @param decayed_block_indices The block indices that were subject to decay
  /// this round.
  /// @return A vector containing the indices of the blocks deallocated.
  std::vector<Index3D> deallocateFullyDecayedBlocks(
      LayerType* layer_ptr, const std::vector<Index3D>& decayed_block_indices);

  // Internal buffers
  host_vector<typename LayerType::BlockType*> allocated_block_ptrs_host_;
  device_vector<typename LayerType::BlockType*> allocated_block_ptrs_device_;
  host_vector<Index3D> allocated_block_indices_host_;
  device_vector<Index3D> allocated_block_indices_device_;
  device_vector<bool> block_fully_decayed_device_;
  host_vector<bool> block_fully_decayed_host_;
};

}  // namespace nvblox

// NOTE(alexmillane): We can't include this CUDA-containing header here, as it
// would expose the user to CUDA. Classes *using* the VoxelDecayer must include
// decay_integrator_impl.cuh themselves.
// #include "nvblox/integrators/internal/impl/decay_integrator_impl.cuh"
