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
#include <nvblox/integrators/internal/decay_integrator.h>

#include "nvblox/integrators/internal/integrators_common.h"

namespace nvblox {

namespace {
template <class LayerType>
std::vector<Index3D> getBlockIndicesToDecay(
    LayerType* layer_ptr, const std::vector<Index3D>& block_indices_to_exclude,
    const std::optional<Vector3f>& exclusion_center,
    const std::optional<float>& exclusion_radius_m) {
  // Create a set so we can do fast lookup of blocks to exclude
  Index3DSet excluded_indices_set;
  excluded_indices_set.insert(block_indices_to_exclude.begin(),
                              block_indices_to_exclude.end());

  const float exclusion_radius_m_sq =
      (exclusion_radius_m) ? (*exclusion_radius_m * *exclusion_radius_m) : -1.0f;

  // Predicate that returns true for blocks we wish to decay
  auto predicate = [&layer_ptr, &excluded_indices_set, &exclusion_center,
                    &exclusion_radius_m_sq](const Index3D& index) {
    if (excluded_indices_set.count(index) == 1) {
      return false;
    }
    if (exclusion_radius_m_sq > 0.0f && exclusion_center) {
      const Vector3f& block_center =
          getPositionFromBlockIndex(layer_ptr->block_size(), index);
      const float dist_sq = (block_center - *exclusion_center).squaredNorm();
      return (dist_sq > exclusion_radius_m_sq);
    }
    return true;
  };

  // Return all indices to decay
  return layer_ptr->getBlockIndicesIf(predicate);
}
}  // namespace

template <class LayerType>
DecayIntegratorBase<LayerType>::DecayIntegratorBase(DecayMode decay_mode) {
  if (decay_mode == DecayMode::kDecayToDeallocate) {
    deallocate_decayed_blocks(true);
  } else if (decay_mode == DecayMode::kDecayToFree) {
    deallocate_decayed_blocks(false);
  } else {
    LOG(FATAL) << "Decay mode not implemented";
  }
}

template <class LayerType>
void DecayIntegratorBase<LayerType>::decay(LayerType* layer_ptr,
                                           const CudaStream cuda_stream) {
  decay(layer_ptr, {}, {}, {}, cuda_stream);
}

template <class LayerType>
void DecayIntegratorBase<LayerType>::decay(
    LayerType* layer_ptr, const std::vector<Index3D>& block_indices_to_exclude,
    const std::optional<Vector3f>& exclusion_center,
    const std::optional<float>& exclusion_radius_m,
    const CudaStream cuda_stream) {
  CHECK_NOTNULL(layer_ptr);

  // Get block indices to decay and their block pointers
  std::vector<Index3D> block_indices_to_decay =
      getBlockIndicesToDecay(layer_ptr, block_indices_to_exclude,
                             exclusion_center, exclusion_radius_m);

  std::vector<typename LayerType::BlockType*> block_ptrs_to_decay =
      getBlockPtrsFromIndices(block_indices_to_decay, layer_ptr);

  if (block_ptrs_to_decay.empty()) {
    // Empty layer, nothing to do here.
    return;
  }

  expandBuffersIfRequired(
      block_ptrs_to_decay.size(), cuda_stream, &allocated_block_ptrs_host_,
      &allocated_block_ptrs_device_, &block_fully_decayed_device_,
      &block_fully_decayed_host_);

  // Get the block pointers on host and copy them to device
  allocated_block_ptrs_host_.copyFromAsync(block_ptrs_to_decay, cuda_stream);
  allocated_block_ptrs_device_.copyFromAsync(allocated_block_ptrs_host_,
                                             cuda_stream);

  block_fully_decayed_device_.resizeAsync(block_ptrs_to_decay.size(),
                                          cuda_stream);

  decayImplementationAsync(layer_ptr, cuda_stream);

  // Copy results back to host and synchronize
  block_fully_decayed_host_.copyFromAsync(block_fully_decayed_device_,
                                          cuda_stream);
  cuda_stream.synchronize();

  // Check if nothing is lost on the way
  CHECK(allocated_block_ptrs_host_.size() == block_ptrs_to_decay.size());
  CHECK(allocated_block_ptrs_device_.size() == block_ptrs_to_decay.size());
  CHECK(block_fully_decayed_device_.size() == block_ptrs_to_decay.size());
  CHECK(block_fully_decayed_host_.size() == block_ptrs_to_decay.size());

  if (deallocate_decayed_blocks_) {
    deallocateFullyDecayedBlocks(layer_ptr, block_indices_to_decay);
  }
}

template <class LayerType>
void DecayIntegratorBase<LayerType>::deallocateFullyDecayedBlocks(
    LayerType* layer_ptr, const std::vector<Index3D>& decayed_block_indices) {
  CHECK(decayed_block_indices.size() == block_fully_decayed_host_.size());

  for (size_t i = 0; i < decayed_block_indices.size(); ++i) {
    if (block_fully_decayed_host_[i]) {
      layer_ptr->clearBlock(decayed_block_indices[i]);
    }
  }
}

template <class LayerType>
bool DecayIntegratorBase<LayerType>::deallocate_decayed_blocks() const {
  return deallocate_decayed_blocks_;
}

template <class LayerType>
void DecayIntegratorBase<LayerType>::deallocate_decayed_blocks(
    bool deallocate_decayed_blocks) {
  deallocate_decayed_blocks_ = deallocate_decayed_blocks;
}

template <class LayerType>
parameters::ParameterTreeNode DecayIntegratorBase<LayerType>::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "decay_integrator" : name_remap;
  return ParameterTreeNode(name,
                           {
                               ParameterTreeNode("deallocate_decayed_blocks:",
                                                 deallocate_decayed_blocks_),
                           });
}

}  // namespace nvblox
