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
#include <string>
#include <vector>

#include "nvblox/core/layer.h"
#include "nvblox/core/types.h"

namespace nvblox {
namespace experiments {

// Note: A good resource for type erasure can be found here:
// https://github.com/alecjacobson/better-code-runtime-polymorphism/blob/master/library.hpp

template <typename LayerType>
bool loadLayer(const std::string& filename, LayerType* layer_ptr) {
  LOG(INFO) << "Loading Layer of type: " << typeid(*layer_ptr).name();
  return true;
}

class LayerInterface {
 public:
  template <typename LayerType>
  LayerInterface(LayerType&& layer)
      : layer_(std::make_unique<LayerModel<LayerType>>(std::move(layer))) {}

  bool load(const std::string& filename) { return layer_->load(filename); }

 private:
  struct LayerConcept {
    virtual ~LayerConcept() = default;
    virtual bool load(const std::string& filename) = 0;
  };

  template <typename LayerType>
  struct LayerModel : LayerConcept {
    LayerModel(LayerType&& layer) : layer_(std::move(layer)) {}

    bool load(const std::string& filename) override {
      return loadLayer(filename, &layer_);
    }

    // Where the layer is actually stored
    LayerType layer_;
  };

  std::unique_ptr<LayerConcept> layer_;
};

class LayerCakeTypeErasure {
 public:
  LayerCakeTypeErasure(float voxel_size, MemoryType memory_type)
      : voxel_size_(voxel_size), memory_type_(memory_type){};

  template <typename LayerType>
  void add() {
    layers_.push_back(LayerType(
        sizeArgumentFromVoxelSize<LayerType>(voxel_size_), memory_type_));
  }

  bool load(const std::string& filename) {
    bool success = true;
    for (auto& layer : layers_) {
      success &= layer.load(filename);
    }
    return success;
  }

 private:
  // Params
  const float voxel_size_;
  const MemoryType memory_type_;

  // Data
  std::vector<LayerInterface> layers_;
};

}  // namespace experiments
}  // namespace nvblox