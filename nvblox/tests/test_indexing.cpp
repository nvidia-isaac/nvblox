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

#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/map/layer.h"
#include "nvblox/map/voxels.h"

#include "nvblox/tests/gpu_indexing.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-4;

class IndexingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::srand(0);
    block_size_ = VoxelBlock<bool>::kVoxelsPerSide * voxel_size_;
  }

  float block_size_;
  float voxel_size_ = 0.05;
};

TEST_F(IndexingTest, CenterIndexing) {
  AlignedVector<Vector3f> test_points;
  test_points.push_back({0, 0, 0});
  test_points.push_back({0.05, 0.05, 0.05});
  test_points.push_back({0.025, 0.025, 0.025});
  test_points.push_back({-1, -3, 2});

  for (const Vector3f& point : test_points) {
    Index3D block_index;
    Index3D voxel_index;
    getBlockAndVoxelIndexFromPositionInLayer(block_size_, point, &block_index,
                                             &voxel_index);

    Vector3f reconstructed_point = getPositionFromBlockIndexAndVoxelIndex(
        block_size_, block_index, voxel_index);
    Vector3f reconstructed_center_point =
        getCenterPositionFromBlockIndexAndVoxelIndex(block_size_, block_index,
                                                     voxel_index);

    // Check the difference between the corner and the center.
    Vector3f center_difference =
        reconstructed_center_point - reconstructed_point;

    // Needs to be within voxel size.
    EXPECT_LT((reconstructed_point - point).cwiseAbs().maxCoeff(), voxel_size_);
    EXPECT_LT((reconstructed_center_point - point).cwiseAbs().maxCoeff(),
              voxel_size_);
    EXPECT_TRUE(center_difference.isApprox(
        Vector3f(voxel_size_ / 2.0f, voxel_size_ / 2.0f, voxel_size_ / 2.0f),
        kFloatEpsilon));
  }
}

TEST_F(IndexingTest, getBlockAndVoxelIndexFromPositionInLayer) {
  constexpr int kNumTests = 1000;
  for (int i = 0; i < kNumTests; i++) {
    // Random block and voxel indices
    constexpr int kRandomBlockIndexRange = 1000;
    const Index3D block_index = test_utils::getRandomIndex3dInRange(
        -kRandomBlockIndexRange, kRandomBlockIndexRange);
    const Index3D voxel_index = test_utils::getRandomIndex3dInRange(
        0, VoxelBlock<bool>::kVoxelsPerSide - 1);

    // 3D point from indices, including sub-voxel randomness
    constexpr float voxel_size = 0.1;
    constexpr float block_size = VoxelBlock<bool>::kVoxelsPerSide * voxel_size;
    const Vector3f delta =
        test_utils::getRandomVector3fInRange(0.0f, voxel_size);
    const Vector3f p_L = (block_index.cast<float>() * block_size) +
                         (voxel_index.cast<float>() * voxel_size) + delta;

    // Point back to voxel indices
    Index3D block_index_check;
    Index3D voxel_index_check;
    getBlockAndVoxelIndexFromPositionInLayer(
        block_size, p_L, &block_index_check, &voxel_index_check);

    // Check we get out what we put in
    EXPECT_EQ(block_index_check.x(), block_index.x());
    EXPECT_EQ(block_index_check.y(), block_index.y());
    EXPECT_EQ(block_index_check.z(), block_index.z());
  }
}

TEST_F(IndexingTest, getBlockAndVoxelIndexFromPositionInLayerRoundingErrors) {
  constexpr int kNumTests = 1e7;
  std::vector<Vector3f> positions;
  positions.reserve(kNumTests);
  for (int i = 0; i < kNumTests; i++) {
    positions.push_back(Vector3f::Random());
  }

  std::vector<Index3D> block_indices;
  std::vector<Index3D> voxel_indices;
  constexpr float kBlockSize = 0.1;
  test_utils::getBlockAndVoxelIndexFromPositionInLayerOnGPU(
      kBlockSize, positions, &block_indices, &voxel_indices);

  for (int i = 0; i < kNumTests; i++) {
    EXPECT_TRUE((voxel_indices[i].array() >= 0).all());
    EXPECT_TRUE(
        (voxel_indices[i].array() < VoxelBlock<bool>::kVoxelsPerSide).all());
  }
}

TEST_F(IndexingTest, BlockCenter) {
  const float kBlockSize = 1.0f;
  const Vector3f block_center =
      getCenterPositionFromBlockIndex(kBlockSize, Index3D::Zero());
  constexpr float kEps = 1e-6;
  EXPECT_NEAR(block_center.x(), 0.5f, kEps);
  EXPECT_NEAR(block_center.y(), 0.5f, kEps);
  EXPECT_NEAR(block_center.z(), 0.5f, kEps);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
