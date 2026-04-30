/*
Copyright 2026 NVIDIA CORPORATION

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

#include <type_traits>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/renderer/utils/push_constants.h"

namespace nvblox {
namespace renderer {
namespace test {

// ==============================================================================
// Push Constants Tests - Verify struct sizes match shader expectations
// ==============================================================================

class PushConstantsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test MeshPushConstants structure
TEST_F(PushConstantsTest, MeshPushConstantsSize) {
  // mat4 (64) + hasTexture uint (4) + reserved[3] (12) = 80 bytes
  EXPECT_EQ(sizeof(MeshPushConstants), 80u);
  EXPECT_EQ(sizeof(MeshPushConstants::viewProj), 64u);
}

// Test PointCloudPushConstants structure
TEST_F(PushConstantsTest, PointCloudPushConstantsSize) {
  // mat4 (64) + float (4) + padding (12) = 80 bytes
  EXPECT_EQ(sizeof(PointCloudPushConstants), 80u);

  // Verify offset of pointSize (should be at byte 64)
  PointCloudPushConstants pc{};
  const auto* base = reinterpret_cast<const char*>(&pc);
  const auto* point_size = reinterpret_cast<const char*>(&pc.pointSize);
  EXPECT_EQ(point_size - base, 64);
}

// Test ImagePushConstants structure
TEST_F(PushConstantsTest, ImagePushConstantsSize) {
  // 4 floats = 16 bytes
  EXPECT_EQ(sizeof(ImagePushConstants), 16u);

  // Verify layout
  ImagePushConstants pc{};
  const auto* base = reinterpret_cast<const char*>(&pc);
  EXPECT_EQ(reinterpret_cast<const char*>(&pc.minDepth) - base, 0);
  EXPECT_EQ(reinterpret_cast<const char*>(&pc.maxDepth) - base, 4);
  EXPECT_EQ(reinterpret_cast<const char*>(&pc.colormap) - base, 8);
  EXPECT_EQ(reinterpret_cast<const char*>(&pc.displayLayout) - base, 12);
}

// Test push constants are standard layout (required for Vulkan interop)
TEST_F(PushConstantsTest, StandardLayout) {
  EXPECT_TRUE(std::is_standard_layout_v<MeshPushConstants>);
  EXPECT_TRUE(std::is_standard_layout_v<PointCloudPushConstants>);
  EXPECT_TRUE(std::is_standard_layout_v<ImagePushConstants>);
}

}  // namespace test
}  // namespace renderer
}  // namespace nvblox

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
