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
#include <gtest/gtest.h>

#include "nvblox/core/blox.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/voxels.h"

using namespace nvblox;

struct NewVoxel {
 float voxel_data = 0.0f;
};

struct NewBlock {
  typedef unified_ptr<NewBlock> Ptr;
  typedef unified_ptr<const NewBlock> ConstPtr;

  static Ptr allocate(MemoryType memory_type) {
    return make_unified<NewBlock>();
  };

  float block_data = 0.0f;
};

using NewVoxelLayer = VoxelBlockLayer<NewVoxel>;
using NewBlockLayer = BlockLayer<NewBlock>;

TEST(TraitsTest, LayerTraits) {
  // Existing layer types
  EXPECT_TRUE(traits::is_voxel_layer<TsdfLayer>());
  EXPECT_FALSE(traits::is_voxel_layer<MeshLayer>());
  EXPECT_TRUE(traits::is_voxel_layer<NewVoxelLayer>());
  EXPECT_FALSE(traits::is_voxel_layer<NewBlockLayer>());
  // New layer types
  static_assert(traits::is_voxel_layer<TsdfLayer>());
  static_assert(!traits::is_voxel_layer<MeshLayer>());
  static_assert(traits::is_voxel_layer<NewVoxelLayer>());
  static_assert(!traits::is_voxel_layer<NewBlockLayer>());
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
