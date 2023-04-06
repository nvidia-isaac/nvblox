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

#include "nvblox/core/internal/error_check.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/utils/timing.h"

namespace nvblox {

template <typename BlockType>
GPULayerView<BlockType>::GPULayerView(LayerType* layer_ptr)
    : gpu_hash_ptr_(std::make_shared<GPUHashImpl<BlockType>>()) {
  reset(layer_ptr);
}

template <typename BlockType>
GPULayerView<BlockType>::GPULayerView(size_t max_num_blocks)
    : gpu_hash_ptr_(std::make_shared<GPUHashImpl<BlockType>>(max_num_blocks)) {
  // The GPUHashImpl takes care of allocating GPU memory.
}

template <typename BlockType>
GPULayerView<BlockType>::~GPULayerView() {
  // The GPUHashImpl takes care of cleaning up GPU memory.
}

template <typename BlockType>
GPULayerView<BlockType>::GPULayerView(const GPULayerView<BlockType>& other)
    : block_size_(other.block_size_), gpu_hash_ptr_(other.gpu_hash_ptr_) {}

template <typename BlockType>
GPULayerView<BlockType>::GPULayerView(GPULayerView<BlockType>&& other)
    : block_size_(other.block_size_),
      gpu_hash_ptr_(std::move(other.gpu_hash_ptr_)) {}

template <typename BlockType>
GPULayerView<BlockType>& GPULayerView<BlockType>::operator=(
    const GPULayerView<BlockType>& other) {
  block_size_ = other.block_size_;
  gpu_hash_ptr_ = other.gpu_hash_ptr_;
  return *this;
}

template <typename BlockType>
GPULayerView<BlockType>& GPULayerView<BlockType>::operator=(
    GPULayerView<BlockType>&& other) {
  block_size_ = other.block_size_;
  gpu_hash_ptr_ = std::move(other.gpu_hash_ptr_);
  return *this;
}

template <typename BlockType>
void GPULayerView<BlockType>::reset(LayerType* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);
  timing::Timer timer("gpu_hash/transfer");

  // Allocate gpu hash if not allocated already
  if (!gpu_hash_ptr_) {
    gpu_hash_ptr_ = std::make_shared<GPUHashImpl<BlockType>>(
        layer_ptr->numAllocatedBlocks() * size_expansion_factor_);
  }

  // Check the load factor and increase the size if required.
  const float current_load_factor =
      static_cast<float>(layer_ptr->numAllocatedBlocks()) /
      static_cast<float>(gpu_hash_ptr_->max_num_blocks_);
  if (current_load_factor > max_load_factor_) {
    const size_t new_max_num_blocks = static_cast<size_t>(std::ceil(
        size_expansion_factor_ * std::max(gpu_hash_ptr_->max_num_blocks_,
                                          layer_ptr->numAllocatedBlocks())));
    VLOG(3) << "Resizing from " << gpu_hash_ptr_->max_num_blocks_ << " to "
            << new_max_num_blocks << " to accomodate "
            << layer_ptr->numAllocatedBlocks();
    reset(new_max_num_blocks);
    CHECK_LT(layer_ptr->numAllocatedBlocks(), gpu_hash_ptr_->max_num_blocks_);
  }

  gpu_hash_ptr_->impl_.clear();

  // This is necessary for bug-free operation, as clear does not sync
  // afterwards.
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
  if (gpu_hash_ptr_->impl_.full()) {
    LOG(ERROR) << "Have a full GPU hash! This is bad!";
  }

  block_size_ = layer_ptr->block_size();

  // Arange blocks in continuous host memory for transfer
  thrust::host_vector<IndexBlockPair<BlockType>> host_block_vector;
  host_block_vector.reserve(layer_ptr->numAllocatedBlocks());
  for (const auto& index : layer_ptr->getAllBlockIndices()) {
    host_block_vector.push_back(IndexBlockPair<BlockType>(
        index, layer_ptr->getBlockAtIndex(index).get()));
  }

  // CPU -> GPU
  thrust::device_vector<IndexBlockPair<BlockType>> device_block_vector(
      host_block_vector);

  // GPU Insert
  // NOTE(alexmillane): We have to do some unfortunate casting here. The problem
  // is that the hash stores pairs with const keys, but the IndexBlockPair
  // vector CANNOT have const keys. So the only way is to perform this cast.
  ConstIndexBlockPair<BlockType>* block_start_raw_ptr =
      reinterpret_cast<ConstIndexBlockPair<BlockType>*>(
          device_block_vector.data().get());
  gpu_hash_ptr_->impl_.insert(
      stdgpu::make_device(block_start_raw_ptr),
      stdgpu::make_device(block_start_raw_ptr + device_block_vector.size()));

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
}

template <typename BlockType>
void GPULayerView<BlockType>::reset(size_t new_max_num_blocks) {
  timing::Timer timer("gpu_hash/transfer/reallocation");
  gpu_hash_ptr_ = std::make_shared<GPUHashImpl<BlockType>>(new_max_num_blocks);
}

template <typename BlockType>
size_t GPULayerView<BlockType>::size() const {
  return gpu_hash_ptr_->size();
}

}  // namespace nvblox