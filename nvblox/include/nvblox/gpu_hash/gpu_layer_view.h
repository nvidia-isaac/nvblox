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

namespace nvblox {

template <typename BlockType>
class GPUHashImpl;

template <typename BlockType>
class BlockLayer;

template <typename BlockType>
class GPULayerView {
 public:
  using LayerType = BlockLayer<BlockType>;

  GPULayerView() = default;
  GPULayerView(size_t max_num_blocks);
  GPULayerView(LayerType* layer_ptr);

  GPULayerView(const GPULayerView& other);
  GPULayerView(GPULayerView&& other);
  GPULayerView& operator=(const GPULayerView& other);
  GPULayerView& operator=(GPULayerView&& other);

  ~GPULayerView();

  // Creates a new GPULayerView from a layer
  void reset(LayerType* layer_ptr);

  // Resizes the underlying GPU hash as well as deleting its contents
  void reset(size_t new_max_num_blocks);

  float block_size() const { return block_size_; }

  const GPUHashImpl<BlockType>& getHash() const { return *gpu_hash_ptr_; }

  size_t size() const;

  // Accessors
  float max_load_factor() const { return max_load_factor_; }
  float size_expansion_factor() const { return size_expansion_factor_; }

 private:
  // Layer params
  float block_size_;

  // The load factor at which we reallocate space. Load factors of above 0.5
  // seem to cause the hash table to overfill in some cases, so please use
  // max loads lower than that.
  const float max_load_factor_ = 0.5;

  // This is the factor by which we overallocate space
  const float size_expansion_factor_ = 2.0f;

  // NOTE(alexmillane): To keep GPU code out of the header we use PIMPL to hide
  // the details of the GPU hash.
  std::shared_ptr<GPUHashImpl<BlockType>> gpu_hash_ptr_;
};

}  // namespace nvblox

// NOTE(alexmillane):
// - I am leaving this here as a reminder to NOT include the implementation.
// - The implementation file is instead included by .cu files which declare
//   specializations.
// - The problem is that we don't want the GPULayerView implementation, which
//   contains CUDA calls and stdgpu code, bleeding into into layer.h, one of our
//   main interace headers.
//#include "nvblox/gpu_hash/internal/cuda/impl/gpu_layer_view_impl.cuh"
