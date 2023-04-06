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

#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/map/layer.h"
#include "nvblox/tests/voxels.h"

namespace nvblox {

// Dummy block type that just stores its own index.
struct IndexBlock {
  typedef unified_ptr<IndexBlock> Ptr;
  typedef unified_ptr<const IndexBlock> ConstPtr;

  Index3D data;

  static Ptr allocate(MemoryType memory_type) {
    return make_unified<IndexBlock>(memory_type);
  }
};

// Dummy block that stores a single bool
struct BooleanBlock {
  typedef unified_ptr<BooleanBlock> Ptr;
  typedef unified_ptr<const BooleanBlock> ConstPtr;

  bool data = false;

  static Ptr allocate(MemoryType memory_type) {
    return make_unified<BooleanBlock>(memory_type);
  }
};

struct FloatBlock {
  typedef unified_ptr<FloatBlock> Ptr;
  typedef unified_ptr<const FloatBlock> ConstPtr;

  static Ptr allocate(MemoryType memory_type) {
    return make_unified<FloatBlock>();
  };

  float block_data = 0.0f;
};

using IndexBlockLayer = BlockLayer<IndexBlock>;
using BooleanBlockLayer = BlockLayer<BooleanBlock>;
using FloatBlockLayer = BlockLayer<FloatBlock>;
using FloatVoxelBlock = VoxelBlock<FloatVoxel>;
using FloatVoxelLayer = VoxelBlockLayer<FloatVoxel>;

}  // namespace nvblox
