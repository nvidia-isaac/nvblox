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
#include "nvblox/utils/logging.h"

#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/rays/ray_caster.h"
#include "nvblox/utils/timing.h"

using namespace nvblox;

class RayCasterTest : public ::testing::Test {
 protected:
  void SetUp() override {
    timing::Timing::Reset();
    std::srand(0);
  }

  static constexpr float kFloatEpsilon = 1e-4;
};

TEST_F(RayCasterTest, StraightAheadCast) {
  Vector3f start(0.0f, 0.0f, 0.0f);
  Vector3f end(5.0f, 0.0f, 0.0f);
  RayCaster raycaster(start, end);

  std::vector<Index3D> all_indices;
  raycaster.getAllIndices(&all_indices);

  EXPECT_EQ(all_indices.size(), 6);

  // Next make a scaled raycaster.
  float scale = 2.0f;
  RayCaster scaled_raycaster(scale * start, scale * end, scale);
  std::vector<Index3D> all_scaled_indices;
  scaled_raycaster.getAllIndices(&all_scaled_indices);

  ASSERT_EQ(all_indices.size(), all_scaled_indices.size());

  // Finally, negative.
  RayCaster negative_raycaster(-start, -end);
  std::vector<Index3D> all_negative_indices;
  negative_raycaster.getAllIndices(&all_negative_indices);

  ASSERT_EQ(all_indices.size(), all_negative_indices.size());

  for (size_t i = 0; i < all_indices.size(); i++) {
    EXPECT_EQ(all_indices[i], all_scaled_indices[i]);
    EXPECT_EQ(all_indices[i], -all_negative_indices[i]);
    EXPECT_EQ(all_indices[i].y(), 0);
    EXPECT_EQ(all_indices[i].z(), 0);
  }
}

TEST_F(RayCasterTest, ObliqueCast) {
  Vector3f start(0.5f, -1.1f, 3.1f);
  Vector3f end(5.1f, 0.2f, 2.1f);
  RayCaster raycaster(start, end);

  std::vector<Index3D> all_indices;
  raycaster.getAllIndices(&all_indices);

  // Next make a scaled raycaster.
  float scale = 2.0f;
  RayCaster scaled_raycaster(scale * start, scale * end, scale);
  std::vector<Index3D> all_scaled_indices;
  scaled_raycaster.getAllIndices(&all_scaled_indices);

  ASSERT_EQ(all_indices.size(), all_scaled_indices.size());

  // Finally, backwards.
  RayCaster backwards_raycaster(end, start);
  std::vector<Index3D> all_backwards_indices;
  backwards_raycaster.getAllIndices(&all_backwards_indices);

  ASSERT_EQ(all_indices.size(), all_backwards_indices.size());

  for (size_t i = 0; i < all_indices.size(); i++) {
    EXPECT_EQ(all_indices[i], all_scaled_indices[i]);
    EXPECT_EQ(all_indices[i],
              all_backwards_indices[all_indices.size() - i - 1]);
  }
}

TEST_F(RayCasterTest, Length0Cast) {
  Vector3f start(0.0f, 0.0f, 0.0f);
  Vector3f end = start;
  RayCaster raycaster(start, end);

  std::vector<Index3D> all_indices;
  raycaster.getAllIndices(&all_indices);

  EXPECT_EQ(all_indices.size(), 1);
  EXPECT_EQ(all_indices[0], Index3D::Zero());
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}