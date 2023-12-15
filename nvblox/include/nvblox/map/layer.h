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
#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/hash.h"
#include "nvblox/core/traits.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"
#include "nvblox/map/blox.h"

namespace nvblox {

/// Base class for all layer objects.
class BaseLayer {
 public:
  typedef std::shared_ptr<BaseLayer> Ptr;
  typedef std::shared_ptr<const BaseLayer> ConstPtr;

  virtual ~BaseLayer() = default;

  // Just an interface class
};

/// A layer that contains blocks, which are stored in a hash map.
template <typename _BlockType>
class BlockLayer : public BaseLayer {
 public:
  typedef std::shared_ptr<BlockLayer> Ptr;
  typedef std::shared_ptr<const BlockLayer> ConstPtr;

  /// Check that custom block types implement allocate functions
  static_assert(traits::has_allocate<_BlockType>::value,
                "BlockType must implement an allocate() function.");

  // FIXME:  fix trait.h and enable static assert
  //  static_assert(traits::has_allocate_async<_BlockType>::value,
  //                "BlockType must implement an allocateAsync() function.");

  /// Allows inspection of the contained BlockType through LayerType::BlockType
  typedef _BlockType BlockType;
  typedef BlockLayer<BlockType> LayerType;
  typedef GPULayerView<BlockType> GPULayerViewType;

  typedef typename Index3DHashMapType<typename BlockType::Ptr>::type BlockHash;

  BlockLayer() = delete;
  BlockLayer(float block_size, MemoryType memory_type)
      : block_size_(block_size),
        memory_type_(memory_type),
        gpu_layer_view_up_to_date_(false) {}
  virtual ~BlockLayer() {}

  /// Use copyFrom() instead of copy constructors
  BlockLayer(const BlockLayer& other) = delete;
  BlockLayer(const BlockLayer& other, MemoryType memory_type) = delete;
  BlockLayer& operator=(const BlockLayer& other) = delete;

  /// Move operations
  BlockLayer(BlockLayer&& other) = default;
  BlockLayer& operator=(BlockLayer&& other) = default;

  /// Deep copies, with optionally changing the memory type.
  void copyFrom(const BlockLayer& other);
  void copyFromAsync(const BlockLayer& other, const CudaStream cuda_stream);

  /// Block accessors by index.
  typename BlockType::Ptr getBlockAtIndex(const Index3D& index);
  typename BlockType::ConstPtr getBlockAtIndex(const Index3D& index) const;
  typename BlockType::Ptr allocateBlockAtIndex(const Index3D& index);
  typename BlockType::Ptr allocateBlockAtIndexAsync(
      const Index3D& index, const CudaStream& cuda_stream);
  void allocateBlocksAtIndices(const std::vector<Index3D>& indices,
                               const CudaStream& cuda_stream);

  /// Block accessors by position.
  typename BlockType::Ptr getBlockAtPosition(const Vector3f& position);
  typename BlockType::ConstPtr getBlockAtPosition(
      const Vector3f& position) const;
  typename BlockType::Ptr allocateBlockAtPositionAsync(
      const Vector3f& position, const CudaStream& cuda_stream);
  typename BlockType::Ptr allocateBlockAtPosition(const Vector3f& position);

  /// Get all blocks indices or pointers.
  std::vector<Index3D> getAllBlockIndices() const;
  std::vector<BlockType*> getAllBlockPointers();

  /// Get block indices for which the provided predicate evaluates to true
  std::vector<Index3D> getBlockIndicesIf(
      std::function<bool(const Index3D&)> predicate);

  /// Check if allocated
  bool isBlockAllocated(const Index3D& index) const;

  __host__ __device__ float block_size() const { return block_size_; }
  int numAllocatedBlocks() const { return blocks_.size(); }
  size_t size() const { return blocks_.size(); }

  /// Clear the layer of all data
  void clear() { blocks_.clear(); }

  // Clear (deallocate) a single block
  bool clearBlock(const Index3D& index);

  /// Clear (deallocate) blocks passed in
  /// Note if a block does not exist, this function just (silently)
  /// continues trying the rest of the list.
  void clearBlocks(const std::vector<Index3D>& indices);

  MemoryType memory_type() const { return memory_type_; }

  /// GPU Hash
  /// Note(alexmillane): The hash returned here is invalidated by calls to
  /// allocateBlock
  GPULayerViewType getGpuLayerView() const;

 protected:
  float block_size_;
  MemoryType memory_type_;

  /// CPU Hash (Index3D -> BlockType::Ptr)
  BlockHash blocks_;

  /// GPU Hash
  /// NOTE(alexmillane):
  /// - This has is subservient to the CPU version. The layer has to copy the
  ///   hash to GPU when it is requested.
  /// - Cached such that if no blocks are allocated between requests, the
  ///   GPULayerView is not recopied.
  /// - Lazily allocated (space allocated on the GPU first request)
  /// - The "mutable" here is to enable caching in const member functions.
  mutable bool gpu_layer_view_up_to_date_;
  mutable std::unique_ptr<GPULayerViewType> gpu_layer_view_;
};

/// Specialization for BlockLayer that exclusively contains VoxelBlocks to make
/// access easier.
template <typename VoxelType>
class VoxelBlockLayer : public BlockLayer<VoxelBlock<VoxelType>> {
 public:
  typedef std::shared_ptr<VoxelBlockLayer> Ptr;
  typedef std::shared_ptr<const VoxelBlockLayer> ConstPtr;

  using Base = BlockLayer<VoxelBlock<VoxelType>>;

  using VoxelBlockType = VoxelBlock<VoxelType>;

  /// Constructor
  /// @param voxel_size The size of each voxel
  /// @param memory_type In which type of memory the blocks in this layer should
  ///                    be stored.
  VoxelBlockLayer(float voxel_size, MemoryType memory_type)
      : BlockLayer<VoxelBlockType>(VoxelBlockType::kVoxelsPerSide * voxel_size,
                                   memory_type),
        voxel_size_(voxel_size) {}
  VoxelBlockLayer() = delete;
  virtual ~VoxelBlockLayer() {}

  /// Deep copies
  VoxelBlockLayer(const VoxelBlockLayer& other);
  VoxelBlockLayer(const VoxelBlockLayer& other, MemoryType memory_type);
  /// Assignment retains the current layer's memory type.
  VoxelBlockLayer& operator=(const VoxelBlockLayer& other);

  /// Move operations
  VoxelBlockLayer(VoxelBlockLayer&& other) = default;
  VoxelBlockLayer& operator=(VoxelBlockLayer&& other) = default;

  /// Gets voxels by copy from a list of positions.
  /// The positions are given with respect to the layer frame (L). The function
  /// returns the closest voxels to the passed points.
  /// If memory_type_ == kDevice, the function retrieves voxel data from the GPU
  /// and transfers it to the CPU. Modifications to the returned voxel data do
  /// not affect the layer (they're copies).
  /// Note that this function performs a Cudamemcpy per voxel. So it will likely
  /// be relatively slow.
  /// @param positions_L query positions in layer frame
  /// @param voxels_ptr a pointer to a vector of voxels where we'll store the
  ///                   output
  /// @param success_flags_ptr a pointer to a vector of flags indicating if we
  ///                          were able to retrive each voxel.
  void getVoxels(const std::vector<Vector3f>& positions_L,
                 std::vector<VoxelType>* voxels_ptr,
                 std::vector<bool>* success_flags_ptr) const;

  /// Gets voxels by copy from a list of positions.
  /// See getVoxels() above. This stream performs the same functionality
  /// except that the copy is performed on a specific CUDA stream.
  void getVoxels(const std::vector<Vector3f>& positions_L,
                 std::vector<VoxelType>* voxels_ptr,
                 std::vector<bool>* success_flags_ptr,
                 CudaStream* cuda_stream_ptr) const;

  /// Gets voxels by copy from a list of positions.
  /// See getVoxels(). This function copies voxels to device vectors.
  /// @param positions_L query positions in layer frame
  /// @param voxels_ptr a pointer to a GPU vector of voxels where we'll store
  ///                   the output
  /// @param success_flags_ptr a pointer to a GPU vector of flags indicating if
  ///                          we were able to retrive each voxel.
  void getVoxelsGPU(const device_vector<Vector3f>& positions_L,
                    device_vector<VoxelType>* voxels_ptr,
                    device_vector<bool>* success_flags_ptr) const;

  /// Gets voxels by copy from a list of positions.
  /// See getVoxelsGPU(). This stream performs the same functionality
  /// except that the copy is performed on a specific CUDA stream.
  void getVoxelsGPU(const device_vector<Vector3f>& positions_L,
                    device_vector<VoxelType>* voxels_ptr,
                    device_vector<bool>* success_flags_ptr,
                    CudaStream* cuda_stream_ptr) const;

  /// Get a voxel by copy by (closest) position
  /// The position is given with respect to the layer frame (L). The function
  /// returns the closest voxels to the passed points.
  /// If memory_type_ == kDevice, the function retrieves voxel data from the GPU
  /// and transfers it to the CPU. Modifications to the returned voxel data do
  /// not affect the layer (they're copies).
  /// Note that this function performs a Cudamemcpy for the voxel. So it's slow.
  /// This function is intended for testing/convenience and shouldn't be used in
  /// performance critical code.
  /// @param p_L query position in layer frame
  /// @return A pair containing the voxel copy and a flag indicating if the
  /// voxel could be retrieved (ie if the voxel was allocated in the layer).
  std::pair<VoxelType, bool> getVoxel(const Vector3f& p_L) const;

  /// Returns the size of the voxels in this layer.
  float voxel_size() const { return voxel_size_; }

 private:
  float voxel_size_;
};

namespace traits {

// Helpers for detecting if a type is a layer.
template <typename Type>
struct is_layer {
  static constexpr bool value = std::is_base_of<BaseLayer, Type>::value;
};

template <typename... Args>
struct are_layers {
  static constexpr bool value = (is_layer<Args>::value && ...);
};

template <typename LayerType>
struct is_voxel_layer : public std::false_type {};

template <typename VoxelType>
struct is_voxel_layer<VoxelBlockLayer<VoxelType>> : public std::true_type {};

}  // namespace traits

// Returns voxel size or block size based on the layer type (at compile time)
template <typename LayerType>
constexpr float sizeArgumentFromVoxelSize(float voxel_size);

}  // namespace nvblox

#include "nvblox/map/internal/impl/layer_impl.h"
