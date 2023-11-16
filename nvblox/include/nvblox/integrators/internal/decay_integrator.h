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
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/core/parameter_tree.h"

namespace nvblox {

// The mode of the decay integrator. Do we decay to deallocated or free.
enum class DecayMode { kDecayToDeallocate, kDecayToFree };
constexpr DecayMode kDefaultDecayMode = DecayMode::kDecayToDeallocate;

/// The DecayIntegrator can be used to decay (reduce the weights) of a layer.
/// This is useful for dynamic scenarios, where object might move/disappear
/// when not constantly observed.
template <class LayerType>
class DecayIntegratorBase {
 public:
  explicit DecayIntegratorBase(DecayMode decay_mode = kDefaultDecayMode);
  virtual ~DecayIntegratorBase() = default;

  DecayIntegratorBase(const DecayIntegratorBase&) = delete;
  DecayIntegratorBase& operator=(const DecayIntegratorBase&) const = delete;
  DecayIntegratorBase(DecayIntegratorBase&&) = delete;
  DecayIntegratorBase& operator=(const DecayIntegratorBase&&) const = delete;

  /// Decay all blocks. Fully decayed blocks (weight close to zero) will be
  /// deallocated if deallocate_decayed_blocks is true.
  ///
  /// @param layer_ptr    Layer to decay
  /// @param cuda_stream  Cuda stream for GPU work
  void decay(LayerType* layer_ptr, const CudaStream cuda_stream);

  /// Decay blocks. Fully decayed blocks (weight close to zero) will be
  /// deallocated if deallocate_decayed_blocks is true. Blocks to decay can be
  /// excluded based on block index and/or distance to point.
  ///
  /// @param layer_ptr                 Layer to decay
  /// @param block_indices_to_exclude  Index of blocks that should not be
  ///                                  decayed
  /// @param exclusion_center          Center for exclusion radius
  /// @param exclusion_radius_m        Exclusion radius
  /// @param cuda_stream               Cuda stream for GPU work
  void decay(LayerType* layer_ptr,
             const std::vector<Index3D>& block_indices_to_exclude,
             const std::optional<Vector3f>& exclusion_center,
             const std::optional<float>& exclusion_radius_m,
             const CudaStream cuda_stream);

  /// A parameter getter
  /// The flag that controls if fully decayed block should be deallocated or
  /// not.
  /// @returns the deallocate_decayed_blocks flag
  bool deallocate_decayed_blocks() const;

  /// A parameter setter
  /// See deallocate_decayed_blocks().
  /// @param deallocate_decayed_blocks the new flag.
  void deallocate_decayed_blocks(bool deallocate_decayed_blocks);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  /// This function should be overriden with the code that updates the weights
  virtual void decayImplementationAsync(LayerType* layer,
                                        const CudaStream cuda_stream) = 0;

  /// Given a vector of blocks that have been decayed, deallocate the ones that
  /// are *fully* decayed (i.e. having a weight that is close to zero)
  void deallocateFullyDecayedBlocks(
      LayerType* layer_ptr, const std::vector<Index3D>& decayed_block_indices);

  // Parameter for the decay step
  bool deallocate_decayed_blocks_ = true;

  // Internal buffers
  host_vector<typename LayerType::BlockType*> allocated_block_ptrs_host_;
  device_vector<typename LayerType::BlockType*> allocated_block_ptrs_device_;
  device_vector<bool> block_fully_decayed_device_;
  host_vector<bool> block_fully_decayed_host_;
};

}  // namespace nvblox

#include "nvblox/integrators/internal/impl/decay_integrator_impl.h"
