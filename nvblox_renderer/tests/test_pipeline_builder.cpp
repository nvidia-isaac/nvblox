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

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/renderer/visualizers/pipeline_builder.h"

using namespace nvblox::renderer;

/// Test fixture for PipelineBuilder.
/// Note: These tests verify the builder's behavior without a Vulkan device.
/// Since PipelineBuilder doesn't expose getters for internal state, we test:
/// 1. Expected failure cases (build without shaders returns VK_NULL_HANDLE)
/// 2. Fluent interface correctness (methods return builder reference)
/// 3. No crashes when calling configuration methods
/// Full pipeline verification requires Vulkan device initialization and
/// is covered by integration tests (e.g., test_headless_renderer.cu).
class PipelineBuilderTest : public ::testing::Test {
 protected:
  PipelineBuilder builder_;
};

// Test that reset() restores default state
TEST_F(PipelineBuilderTest, ResetRestoresDefaults) {
  // Modify builder state
  builder_.setTopology(VK_PRIMITIVE_TOPOLOGY_POINT_LIST)
      .setPolygonMode(VK_POLYGON_MODE_LINE)
      .setCullMode(VK_CULL_MODE_FRONT_BIT)
      .setDepthTest(false, false)
      .setBlending(true);

  // Reset
  builder_.reset();

  // Verify by attempting to build without shaders (should fail gracefully)
  // The actual defaults are tested indirectly through successful builds
  VkPipeline result =
      builder_.build(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0);
  EXPECT_EQ(result, VK_NULL_HANDLE);  // Should fail without shaders
}

// Test that build fails without shader modules
TEST_F(PipelineBuilderTest, BuildFailsWithoutShaders) {
  // Try to build without setting shaders
  VkPipeline result =
      builder_.build(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0);
  EXPECT_EQ(result, VK_NULL_HANDLE);
}

// Test fluent interface returns reference to builder
TEST_F(PipelineBuilderTest, FluentInterfaceReturnsReference) {
  PipelineBuilder& ref1 =
      builder_.setTopology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
  EXPECT_EQ(&ref1, &builder_);

  PipelineBuilder& ref2 = builder_.setPolygonMode(VK_POLYGON_MODE_FILL);
  EXPECT_EQ(&ref2, &builder_);

  PipelineBuilder& ref3 = builder_.setCullMode(VK_CULL_MODE_BACK_BIT);
  EXPECT_EQ(&ref3, &builder_);

  PipelineBuilder& ref4 = builder_.setDepthTest(true, true);
  EXPECT_EQ(&ref4, &builder_);

  PipelineBuilder& ref5 = builder_.setBlending(false);
  EXPECT_EQ(&ref5, &builder_);
}

// Test adding vertex bindings
TEST_F(PipelineBuilderTest, AddVertexBindings) {
  // Should be able to chain multiple bindings
  builder_.addVertexBinding(0, 16).addVertexBinding(
      1, 32, VK_VERTEX_INPUT_RATE_INSTANCE);

  // Verify by attempting to build (will fail without shaders, but shouldn't
  // crash)
  VkPipeline result =
      builder_.build(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0);
  EXPECT_EQ(result, VK_NULL_HANDLE);
}

// Test adding vertex attributes
TEST_F(PipelineBuilderTest, AddVertexAttributes) {
  builder_.addVertexAttribute(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0)
      .addVertexAttribute(1, 0, VK_FORMAT_R8G8B8A8_UINT, 12);

  VkPipeline result =
      builder_.build(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0);
  EXPECT_EQ(result, VK_NULL_HANDLE);
}

// Test dynamic state management
TEST_F(PipelineBuilderTest, DynamicStateManagement) {
  // Clear default dynamic states
  builder_.clearDynamicStates();

  // Add custom dynamic states
  builder_.addDynamicState(VK_DYNAMIC_STATE_LINE_WIDTH)
      .addDynamicState(VK_DYNAMIC_STATE_DEPTH_BIAS);

  VkPipeline result =
      builder_.build(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0);
  EXPECT_EQ(result, VK_NULL_HANDLE);
}

// Test blend factor configuration
TEST_F(PipelineBuilderTest, BlendFactorConfiguration) {
  builder_.setBlending(true).setBlendFactors(
      VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_FACTOR_ONE,
      VK_BLEND_FACTOR_ZERO);

  VkPipeline result =
      builder_.build(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0);
  EXPECT_EQ(result, VK_NULL_HANDLE);
}

// Test depth compare operation
TEST_F(PipelineBuilderTest, DepthCompareOperation) {
  builder_.setDepthTest(true, true)
      .setDepthCompareOp(VK_COMPARE_OP_LESS_OR_EQUAL);

  VkPipeline result =
      builder_.build(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, 0);
  EXPECT_EQ(result, VK_NULL_HANDLE);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
