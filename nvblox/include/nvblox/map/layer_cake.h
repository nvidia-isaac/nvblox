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
#include <typeindex>
#include <unordered_map>

#include "nvblox/core/types.h"
#include "nvblox/map/layer.h"

namespace nvblox {

/// Holds a collection of layers. Currently a restriction that only 1 layer
/// of each type can be stored, and all layers should have the same voxel
/// (or block) size.
class LayerCake {
 public:
  LayerCake() = default;
  LayerCake(float voxel_size) : voxel_size_(voxel_size) {}

  /// Deep Copy (disallowed for now)
  LayerCake(const LayerCake& other) = delete;
  LayerCake& operator=(const LayerCake& other) = delete;

  /// Move
  LayerCake(LayerCake&& other) = default;
  LayerCake& operator=(LayerCake&& other) = default;

  template <typename LayerType>
  LayerType* add(MemoryType memory_type);

  /// Moves the ownership of the layer to the LayerCake.
  inline void insert(const std::type_index& type_index,
                     std::unique_ptr<BaseLayer>&& layer);

  /// Retrieve layers (as pointers)
  template <typename LayerType>
  LayerType* getPtr();
  template <typename LayerType>
  const LayerType* getConstPtr() const;

  /// Retrieve layers (as reference) (will fail if the layer doesn't exist)
  template <typename LayerType>
  const LayerType& get() const;

  template <typename LayerType>
  bool exists() const;

  bool empty() const { return layers_.empty(); }

  void clear() { layers_.clear(); }

  /// Factory accepting a list of LayerTypes (and MemoryTypes)
  template <typename... LayerTypes>
  static LayerCake create(float voxel_size, MemoryType memory_type);
  template <typename... LayerTypes, typename... MemoryTypes>
  static LayerCake create(float voxel_size, MemoryTypes... memory_type);

  const std::unordered_map<std::type_index, std::unique_ptr<BaseLayer>>&
  get_layers() const {
    return layers_;
  }

  float voxel_size() const { return voxel_size_; }
  float block_size() const {
    return voxel_size_ * VoxelBlock<bool>::kVoxelsPerSide;
  }

 private:
  // Params
  float voxel_size_ = 0.0f;

  /// Stored layers
  /// Note(alexmillane): Currently we restrict the cake to storing a single
  /// layer of each type.
  std::unordered_map<std::type_index, std::unique_ptr<BaseLayer>> layers_;
};

}  // namespace nvblox

#include "nvblox/map/internal/impl/layer_cake_impl.h"
