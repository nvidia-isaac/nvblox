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

#include "nvblox/map/accessors.h"
#include "nvblox/map/blox.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"

#include "nvblox/gpu_hash/gpu_layer_view.h"

#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"

#include "nvblox/tests/gpu_layer_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

TEST(GpuHashTest, CopyOverLayer) {
  // Indices to insert in the layer
  std::vector<Index3D> block_indices;
  block_indices.push_back({0, 0, 0});
  block_indices.push_back({1, 0, 0});
  block_indices.push_back({2, 0, 0});

  // Create a layer with a single block
  const float voxel_size = 0.1f;
  TsdfLayer tsdf_layer(voxel_size, MemoryType::kUnified);
  std::for_each(block_indices.begin(), block_indices.end(),
                [&tsdf_layer](const Index3D& idx) {
                  tsdf_layer.allocateBlockAtIndex(idx);
                });

  // Copy block hash to GPU
  GPULayerView<TsdfBlock> gpu_layer = tsdf_layer.getGpuLayerView();

  // Search for indices on the GPU
  block_indices.push_back({3, 0, 0});
  const std::vector<bool> flags =
      test_utils::getContainsFlags(gpu_layer, block_indices);

  EXPECT_EQ(flags.size(), block_indices.size());
  EXPECT_TRUE(flags[0]);
  EXPECT_TRUE(flags[1]);
  EXPECT_TRUE(flags[2]);
  EXPECT_FALSE(flags[3]);
}

TEST(GpuHashTest, SphereSceneAccessTest) {
  // Sphere in a box scene.
  primitives::Scene scene;
  const Vector3f bounds_min = Vector3f(-5.0f, -5.0f, 0.0f);
  const Vector3f bounds_max = Vector3f(5.0f, 5.0f, 5.0f);
  scene.aabb() = AxisAlignedBoundingBox(bounds_min, bounds_max);
  scene.addGroundLevel(0.0f);
  scene.addCeiling(5.0f);
  scene.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0f, 2.0f), 2.0f));
  scene.addPlaneBoundaries(-5.0f, 5.0f, -5.0f, 5.0f);

  // Layer
  constexpr float kVoxelSize = 0.1f;
  TsdfLayer tsdf_layer(kVoxelSize, MemoryType::kUnified);

  // Generate SDF
  constexpr float kMaxDistance = 5.0f;
  scene.generateLayerFromScene(kMaxDistance, &tsdf_layer);

  // Get some random points in the scene
  constexpr int kNumPoints = 1000;
  std::vector<Vector3f> p_L_vec(kNumPoints);
  std::generate(p_L_vec.begin(), p_L_vec.end(), [&bounds_min, &bounds_max]() {
    return test_utils::getRandomVector3fInRange(bounds_min, bounds_max);
  });

  // Lookup voxels (CPU)
  std::vector<TsdfVoxel> cpu_lookup_voxels;
  cpu_lookup_voxels.reserve(p_L_vec.size());
  for (const Vector3f& p_L : p_L_vec) {
    // TODO(alexmillane): Let's also add this as a member function
    const TsdfVoxel* voxel_ptr;
    EXPECT_TRUE(getVoxelAtPosition(tsdf_layer, p_L, &voxel_ptr));
    cpu_lookup_voxels.push_back(*voxel_ptr);
  }

  // CPU -> GPU
  GPULayerView<TsdfBlock> gpu_layer = tsdf_layer.getGpuLayerView();

  // Lookup voxels (GPU)
  std::vector<TsdfVoxel> gpu_lookup_voxels;
  std::vector<bool> flags;
  std::tie(gpu_lookup_voxels, flags) =
      test_utils::getVoxelsAtPositionsOnGPU(gpu_layer, p_L_vec);

  std::for_each(flags.begin(), flags.end(),
                [](bool flag) { EXPECT_TRUE(flag); });

  CHECK_EQ(cpu_lookup_voxels.size(), gpu_lookup_voxels.size());
  for (size_t i = 0; i < gpu_lookup_voxels.size(); i++) {
    constexpr float eps = 1e-6;
    EXPECT_NEAR(cpu_lookup_voxels[i].distance, gpu_lookup_voxels[i].distance,
                eps);
  }

  // Get some points OUTSIDE the scene
  LOG(INFO) << "Testing points outside the scene.";
  std::generate(p_L_vec.begin(), p_L_vec.end(), [&bounds_max]() {
    return test_utils::getRandomVector3fInRange(bounds_max.array() + 1.0,
                                                bounds_max.array() + 5.0);
  });
  std::tie(gpu_lookup_voxels, flags) =
      test_utils::getVoxelsAtPositionsOnGPU(gpu_layer, p_L_vec);
  std::for_each(flags.begin(), flags.end(),
                [](bool flag) { EXPECT_FALSE(flag); });
}

TEST(GpuHashTest, ResizeTest) {
  // Indices to insert in the layer
  std::vector<Index3D> block_indices;
  block_indices.push_back({0, 0, 0});
  block_indices.push_back({1, 0, 0});
  block_indices.push_back({2, 0, 0});

  // Create a layer with a two blocks
  const float voxel_size = 0.1f;
  TsdfLayer tsdf_layer(voxel_size, MemoryType::kUnified);
  std::for_each(block_indices.begin(), block_indices.end(),
                [&tsdf_layer](const Index3D& idx) {
                  tsdf_layer.allocateBlockAtIndex(idx);
                });

  // Push the layer into a undersized layer view and check it expands to fit.
  GPULayerView<TsdfBlock> gpu_layer_view(1);
  gpu_layer_view.reset(&tsdf_layer);
  EXPECT_GT(gpu_layer_view.size(), 1);
}

TEST(GpuHashTest, LoadFactorTest) {
  // Create a layer with 1000 blocks
  constexpr int num_blocks = 1000;
  std::vector<Index3D> block_indices;
  for (int i = 0; i < num_blocks; i++) {
    block_indices.push_back({i, 0, 0});
  }

  // Create a layer
  const float voxel_size = 0.1f;
  TsdfLayer tsdf_layer(voxel_size, MemoryType::kUnified);
  std::for_each(block_indices.begin(), block_indices.end(),
                [&tsdf_layer](const Index3D& idx) {
                  tsdf_layer.allocateBlockAtIndex(idx);
                });

  // Allocate a GPULayerView that can *just* hold the layer without breaking the
  // load rate.
  GPULayerView<TsdfBlock> gpu_layer(1);
  const size_t initial_gpu_layer_size =
      std::ceil(static_cast<float>(num_blocks) / gpu_layer.max_load_factor());
  gpu_layer = GPULayerView<TsdfBlock>(initial_gpu_layer_size);

  // Send in the blocks
  gpu_layer.reset(&tsdf_layer);

  // Check it hasn't expanded
  EXPECT_EQ(gpu_layer.size(), initial_gpu_layer_size);

  // Add another block
  tsdf_layer.allocateBlockAtIndex({0, 0, 1});

  // Regenerate the layer view
  gpu_layer.reset(&tsdf_layer);

  // Check it's expanded (by the right amount)
  EXPECT_GT(gpu_layer.size(), initial_gpu_layer_size);
  EXPECT_NEAR(static_cast<float>(gpu_layer.size()) /
                  static_cast<float>(initial_gpu_layer_size),
              gpu_layer.size_expansion_factor(), 0.01);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
