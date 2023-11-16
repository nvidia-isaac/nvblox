/*
Copyright 2022-2023 NVIDIA CORPORATION

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

#include "nvblox/map/blox.h"
#include "nvblox/map/layer.h"

namespace nvblox {

/// Convert a list of BlockIndices on host, to a list of (non-const) device
/// pointers on host.
///
/// Note that this function will check fail if one passed BlockIndices
/// indices is not allocated. The function is intended for use in specific parts
/// of the code where blocks in the camera view have already been allocated.
/// @param block_indices A vector of the 3D indices of blocks who's pointers we
/// want
/// @param layer_ptr A pointer to the layer containing the blocks.
/// @param return a vector of block pointers.
template <typename BlockType>
__host__ std::vector<BlockType*> getBlockPtrsFromIndices(
    const std::vector<Index3D>& block_indices,
    BlockLayer<BlockType>* layer_ptr);

/// Convert a list of BlockIndices on host, to a list of (const) device pointers
/// on host.
///
/// Note that this function will check fail if one passed BlockIndices
/// indices is not allocated. The function is intended for use in specific parts
/// of the code where blocks in the camera view have already been allocated.
/// @param block_indices A vector of the 3D indices of blocks who's pointers we
/// want
/// @param layer_ptr A pointer to the layer containing the blocks.
/// @param return a vector of block pointers.
template <typename BlockType>
__host__ std::vector<const BlockType*> getBlockPtrsFromIndices(
    const std::vector<Index3D>& block_indices,
    const BlockLayer<BlockType>& layer_ptr);

/// Allocates blocks in the block_indices list which are not already allocated.
/// @param block_indices A vector of the 3D indices of blocks we wanna allocate.
/// @param layer A pointer to the layer where we wanna allocate.
template <typename BlockType>
void allocateBlocksWhereRequired(const std::vector<Index3D>& block_indices,
                                 BlockLayer<BlockType>* layer,
                                 const CudaStream& cuda_stream);

/// Expands a buffers to required_min_size*1.5 factor if available capacity is
/// smaller than required_min_size.
/// @tparam ...Args Type of buffers. Must be unified_vector, host_vector, or
/// device_vector.
/// @param required_min_size The minimum size required of the buffers.
/// @param cuda_stream The stream to do the expansion on
/// @param ...args The unified_vectors.
template <class... Args>
void expandBuffersIfRequired(size_t required_min_size, CudaStream cuda_stream,
                             Args... args);

/// Copies block pointers belonging to a vector of block indices to the
/// device.
///
/// Note: We stage the indices on host pinned memory before copying to
/// the device, as we found it to be faster than copying from system memory
/// directly.
/// @tparam BlockType The type of the blocks in the layer.
/// @param block_indices The block indices to transfer to device.
/// @param cuda_stream The cuda stream to run on.
/// @param layer_ptr The layer in which the blocks are in.
/// @param block_ptrs_host The host pinned memory buffer for the indices.
/// @param block_ptrs_device The device memory buffer for the indices.
template <typename BlockType>
void transferBlockPointersToDevice(
    const std::vector<Index3D>& block_indices, CudaStream cuda_stream,
    BlockLayer<BlockType>* layer_ptr, host_vector<BlockType*>* block_ptrs_host,
    device_vector<BlockType*>* block_ptrs_device);

/// Copies a vector of block indices to the device.
///
/// Note: We stage the indices on host pinned memory before copying
/// to the device, as we found it to be faster than copying from system memory
/// directly.
/// @param block_indices The block indices to transfer to device.
/// @param cuda_stream The cuda stream to run on.
/// @param block_indices_host The host pinned memory buffer for the block
/// pointers.
/// @param block_indices_device  The device memory buffer for the block
/// pointers.
void transferBlocksIndicesToDevice(
    const std::vector<Index3D>& block_indices, CudaStream cuda_stream,
    host_vector<Index3D>* block_indices_host,
    device_vector<Index3D>* block_indices_device);

}  // namespace nvblox

#include "nvblox/integrators/internal/impl/integrators_common_impl.h"
