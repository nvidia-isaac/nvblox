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

#include "nvblox/renderer/core/vk_context.h"
#include "nvblox/renderer/utils/shader_utils.h"

namespace nvblox {
namespace renderer {
namespace test {

class ShaderUtilsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  void TearDown() override {}
};

// Test getShaderDir() returns non-empty string (SHADER_DIR should be defined)
TEST_F(ShaderUtilsTest, GetShaderDir) {
  std::string shader_dir = getShaderDir();
  // SHADER_DIR should be defined at compile time
  EXPECT_FALSE(shader_dir.empty()) << "SHADER_DIR not defined at compile time";
}

// Test readShaderFile() with non-existent file
TEST_F(ShaderUtilsTest, ReadShaderFileNonExistent) {
  std::vector<uint32_t> code = readShaderFile("/non/existent/path.spv");
  EXPECT_TRUE(code.empty());
}

// Test readShaderFile() with empty path
TEST_F(ShaderUtilsTest, ReadShaderFileEmptyPath) {
  std::vector<uint32_t> code = readShaderFile("");
  EXPECT_TRUE(code.empty());
}

// Test ShaderPair default state
TEST_F(ShaderUtilsTest, ShaderPairDefaultState) {
  ShaderPair pair;
  EXPECT_EQ(pair.vert, VK_NULL_HANDLE);
  EXPECT_EQ(pair.frag, VK_NULL_HANDLE);
  EXPECT_FALSE(pair.isValid());
}

// Test ShaderPair::destroy with null handles (should not crash)
TEST_F(ShaderUtilsTest, ShaderPairDestroyNullHandles) {
  ShaderPair pair;
  // Should not crash with null device and null handles
  pair.destroy(VK_NULL_HANDLE);
  EXPECT_FALSE(pair.isValid());
}

// Test createShaderModule() with empty bytecode
TEST_F(ShaderUtilsTest, CreateShaderModuleEmptyCode) {
  std::vector<uint32_t> empty_code;
  VkShaderModule module = createShaderModule(VK_NULL_HANDLE, empty_code);
  EXPECT_EQ(module, VK_NULL_HANDLE);
}

// Test createShaderModule() with null device
TEST_F(ShaderUtilsTest, CreateShaderModuleNullDevice) {
  // Minimal valid SPIR-V header (not a real shader, just for testing)
  std::vector<uint32_t> dummy_code(256, 0);
  VkShaderModule module = createShaderModule(VK_NULL_HANDLE, dummy_code);
  EXPECT_EQ(module, VK_NULL_HANDLE);
}

// =============================================================================
// Integration tests requiring VkContext
// =============================================================================

class ShaderUtilsIntegrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initialize Vulkan context (init() also selects physical device)
    if (!ctx_.init("test_shader_utils", {}, true)) {
      GTEST_SKIP() << "Failed to initialize Vulkan instance";
    }
    if (!ctx_.createDevice()) {
      GTEST_SKIP() << "Failed to create Vulkan device";
    }
  }

  void TearDown() override {
    // VkContext destructor handles cleanup
  }

  VkContext ctx_;
};

// Test readShaderFile() with valid shader files
TEST_F(ShaderUtilsIntegrationTest, ReadValidShaderFile) {
  std::string shader_dir = getShaderDir();
  if (shader_dir.empty()) {
    GTEST_SKIP() << "SHADER_DIR not defined";
  }

  // Try to read mesh.vert.spv (should exist if shaders were compiled)
  std::string vert_path = shader_dir + "/mesh.vert.spv";
  std::vector<uint32_t> code = readShaderFile(vert_path);

  // The file should exist and contain SPIR-V data
  EXPECT_FALSE(code.empty()) << "Could not read shader file: " << vert_path;

  // SPIR-V magic number is 0x07230203
  if (code.size() >= 4) {
    uint32_t magic = *reinterpret_cast<const uint32_t*>(code.data());
    EXPECT_EQ(magic, 0x07230203u) << "Invalid SPIR-V magic number";
  }
}

// Test createShaderModule() with valid bytecode
TEST_F(ShaderUtilsIntegrationTest, CreateValidShaderModule) {
  std::string shader_dir = getShaderDir();
  if (shader_dir.empty()) {
    GTEST_SKIP() << "SHADER_DIR not defined";
  }

  std::string vert_path = shader_dir + "/mesh.vert.spv";
  std::vector<uint32_t> code = readShaderFile(vert_path);
  if (code.empty()) {
    GTEST_SKIP() << "Could not read shader file: " << vert_path;
  }

  VkShaderModule module = createShaderModule(ctx_.device(), code);
  EXPECT_NE(module, VK_NULL_HANDLE);

  // Cleanup
  if (module != VK_NULL_HANDLE) {
    vkDestroyShaderModule(ctx_.device(), module, nullptr);
  }
}

// Test loadShaderPair() with valid shader name
TEST_F(ShaderUtilsIntegrationTest, LoadValidShaderPair) {
  std::string shader_dir = getShaderDir();
  if (shader_dir.empty()) {
    GTEST_SKIP() << "SHADER_DIR not defined";
  }

  ShaderPair pair = loadShaderPair(ctx_.device(), "mesh");
  EXPECT_TRUE(pair.isValid()) << "Failed to load mesh shader pair";

  // Cleanup
  pair.destroy(ctx_.device());
  EXPECT_FALSE(pair.isValid());
}

// Test loadShaderPair() with invalid shader name
TEST_F(ShaderUtilsIntegrationTest, LoadInvalidShaderPair) {
  ShaderPair pair = loadShaderPair(ctx_.device(), "nonexistent_shader");
  EXPECT_FALSE(pair.isValid());
}

// Test loading multiple shader pairs
TEST_F(ShaderUtilsIntegrationTest, LoadMultipleShaderPairs) {
  std::string shader_dir = getShaderDir();
  if (shader_dir.empty()) {
    GTEST_SKIP() << "SHADER_DIR not defined";
  }

  // Load all standard shader pairs
  ShaderPair mesh = loadShaderPair(ctx_.device(), "mesh");
  ShaderPair point_cloud = loadShaderPair(ctx_.device(), "point_cloud");
  ShaderPair image_quad = loadShaderPair(ctx_.device(), "image_quad");

  EXPECT_TRUE(mesh.isValid()) << "Failed to load mesh shaders";
  EXPECT_TRUE(point_cloud.isValid()) << "Failed to load point_cloud shaders";
  EXPECT_TRUE(image_quad.isValid()) << "Failed to load image_quad shaders";

  // Cleanup
  mesh.destroy(ctx_.device());
  point_cloud.destroy(ctx_.device());
  image_quad.destroy(ctx_.device());
}

// Test ShaderPair::destroy with valid handles
TEST_F(ShaderUtilsIntegrationTest, ShaderPairDestroyValidHandles) {
  std::string shader_dir = getShaderDir();
  if (shader_dir.empty()) {
    GTEST_SKIP() << "SHADER_DIR not defined";
  }

  ShaderPair pair = loadShaderPair(ctx_.device(), "mesh");
  if (!pair.isValid()) {
    GTEST_SKIP() << "Could not load mesh shaders";
  }

  EXPECT_NE(pair.vert, VK_NULL_HANDLE);
  EXPECT_NE(pair.frag, VK_NULL_HANDLE);

  pair.destroy(ctx_.device());

  EXPECT_EQ(pair.vert, VK_NULL_HANDLE);
  EXPECT_EQ(pair.frag, VK_NULL_HANDLE);
  EXPECT_FALSE(pair.isValid());
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
