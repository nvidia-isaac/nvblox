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

#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"
#include "nvblox/tests/blox.h"

using namespace nvblox;

TEST(TraitsTest, LayerTraits) {
  // Existing layer types
  EXPECT_TRUE(traits::is_voxel_layer<TsdfLayer>());
  EXPECT_FALSE(traits::is_voxel_layer<MeshLayer>());
  EXPECT_TRUE(traits::is_voxel_layer<FloatVoxelLayer>());
  EXPECT_FALSE(traits::is_voxel_layer<FloatBlockLayer>());
  // New layer types
  static_assert(traits::is_voxel_layer<TsdfLayer>());
  static_assert(!traits::is_voxel_layer<MeshLayer>());
  static_assert(traits::is_voxel_layer<FloatVoxelLayer>());
  static_assert(!traits::is_voxel_layer<FloatBlockLayer>());
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
