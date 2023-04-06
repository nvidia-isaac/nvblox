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
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "nvblox/utils/logging.h"

#include "nvblox/core/types.h"
#include "nvblox/geometry/bounding_spheres.h"

using namespace nvblox;

std::vector<Index3D> get3x3CubeOfBlocks() {
  std::vector<Index3D> block_indices;
  for (int x = -1; x <= 1; x++) {
    for (int y = -1; y <= 1; y++) {
      for (int z = -1; z <= 1; z++) {
        block_indices.push_back(Index3D(x, y, z));
      }
    }
  }
  return block_indices;
}

bool isInResult(const std::vector<Index3D>& result_vec,
                const Index3D& expected_result) {
  for (const Index3D idx : result_vec) {
    if ((idx.array() == expected_result.array()).all()) {
      return true;
    }
  }
  return false;
}

bool isNotInResult(const std::vector<Index3D>& result_vec,
                   const Index3D& expected_non_result) {
  for (const Index3D idx : result_vec) {
    if ((idx.array() == expected_non_result.array()).all()) {
      return false;
    }
  }
  return true;
}

TEST(BoundingSpheresTest, BlocksInside) {
  const std::vector<Index3D> cube_indices = get3x3CubeOfBlocks();

  Vector3f center(0.5f, 0.5f, 0.5f);

  std::vector<Index3D> result =
      getBlocksWithinRadius(cube_indices, 1.0f, center, 0.45f);
  EXPECT_EQ(result.size(), 1);
  EXPECT_TRUE(isInResult(result, Index3D(0, 0, 0)));

  result = getBlocksWithinRadius(cube_indices, 1.0f, center, 0.55f);
  EXPECT_EQ(result.size(), 7);
  EXPECT_TRUE(isInResult(result, Index3D(-1, 0, 0)));
  EXPECT_TRUE(isInResult(result, Index3D(0, -1, 0)));
  EXPECT_TRUE(isInResult(result, Index3D(0, 0, -1)));
  EXPECT_TRUE(isInResult(result, Index3D(0, 0, 0)));
  EXPECT_TRUE(isInResult(result, Index3D(1, 0, 0)));
  EXPECT_TRUE(isInResult(result, Index3D(0, 1, 0)));
  EXPECT_TRUE(isInResult(result, Index3D(0, 0, 1)));

  result =
      getBlocksWithinRadius(cube_indices, 1.0f, center, sqrt(3) / 2 + 0.01);
  EXPECT_EQ(result.size(), 27);
}

TEST(BoundingSpheresTest, BlocksOutside) {
  const std::vector<Index3D> cube_indices = get3x3CubeOfBlocks();

  Vector3f center(0.5f, 0.5f, 0.5f);

  std::vector<Index3D> result =
      getBlocksOutsideRadius(cube_indices, 1.0f, center, 0.45f);
  EXPECT_EQ(result.size(), 26);
  EXPECT_TRUE(isNotInResult(result, Index3D(0, 0, 0)));

  result = getBlocksOutsideRadius(cube_indices, 1.0f, center, 0.55f);
  EXPECT_EQ(result.size(), 20);
  EXPECT_TRUE(isNotInResult(result, Index3D(-1, 0, 0)));
  EXPECT_TRUE(isNotInResult(result, Index3D(0, -1, 0)));
  EXPECT_TRUE(isNotInResult(result, Index3D(0, 0, -1)));
  EXPECT_TRUE(isNotInResult(result, Index3D(0, 0, 0)));
  EXPECT_TRUE(isNotInResult(result, Index3D(1, 0, 0)));
  EXPECT_TRUE(isNotInResult(result, Index3D(0, 1, 0)));
  EXPECT_TRUE(isNotInResult(result, Index3D(0, 0, 1)));

  result =
      getBlocksOutsideRadius(cube_indices, 1.0f, center, sqrt(3) / 2 + 0.01);
  EXPECT_EQ(result.size(), 0);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
