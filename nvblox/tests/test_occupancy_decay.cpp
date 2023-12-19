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

#include "nvblox/core/log_odds.h"
#include "nvblox/integrators/occupancy_decay_integrator.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

// Just do some "random" reproducible index to log odds encoding.
float inline encodeIndexToLogOdds(const Index3D& block_index,
                                  const Index3D& voxel_index,
                                  const float kMaxLogOdds = 1000.f) {
  // Make sure some of them are negative.
  float sign_factor = 1.0;
  if (voxel_index.x() % 2) {
    sign_factor *= -1.0;
  }
  const float value = std::abs((block_index.x() + block_index.z()) * 0.1f +
                               (block_index.y() + voxel_index.x()) +
                               (voxel_index.y() + voxel_index.z()) * 10.f);
  return sign_factor * fmin(value, kMaxLogOdds);
}

OccupancyLayer initLayer(float KVoxelSize, int kNumBlocksToAllocate,
                         float kMaxLogOdds = 1000.f) {
  // Make sure this is deterministic.
  std::srand(0);

  // Empty layer
  OccupancyLayer layer(KVoxelSize, MemoryType::kUnified);

  // Allocate random blocks
  for (int i = 0; i < kNumBlocksToAllocate; i++) {
    layer.allocateBlockAtPosition(
        test_utils::getRandomVector3fInRange(-100.f, 100.f));
  }

  // Init the voxels
  auto init_voxels_lambda = [&kMaxLogOdds](const Index3D& block_index,
                                           const Index3D& voxel_index,
                                           OccupancyVoxel* voxel_ptr) {
    voxel_ptr->log_odds =
        encodeIndexToLogOdds(block_index, voxel_index, kMaxLogOdds);
  };
  callFunctionOnAllVoxels<OccupancyVoxel>(&layer, init_voxels_lambda);
  return layer;
}

TEST(LayerDecayTest, EmptyLayerTest) {
  // Empty layer (check that integrator does not crash)
  constexpr float KVoxelSize = 0.05;
  OccupancyLayer layer(KVoxelSize, MemoryType::kUnified);

  OccupancyDecayIntegrator decay_integrator;
  const std::vector<Index3D> deallocated_blocks =
      decay_integrator.decay(&layer, CudaStreamOwning());

  EXPECT_EQ(layer.numAllocatedBlocks(), 0);
  EXPECT_EQ(deallocated_blocks.size(), 0);
}

TEST(LayerDecayTest, SingleDecayTest) {
  // Test that a single decay does what we would expect.
  constexpr float kEps = 1.0e-6;
  constexpr int kNumBlocksToAllocate = 100;
  constexpr int kNumVoxels =
      kNumBlocksToAllocate * pow(OccupancyBlock::kVoxelsPerSide, 3);
  OccupancyLayer layer = initLayer(0.05f, kNumBlocksToAllocate);

  // Decay the occupancy probabilities
  OccupancyDecayIntegrator decay_integrator;
  decay_integrator.deallocate_decayed_blocks(false);
  const std::vector<Index3D> deallocated_blocks =
      decay_integrator.decay(&layer, CudaStreamOwning());
  EXPECT_EQ(deallocated_blocks.size(), 0);

  // Check if this worked
  int num_checked_voxels = 0;
  auto check_decay_lambda = [&decay_integrator, &num_checked_voxels](
                                const Index3D& block_index,
                                const Index3D& voxel_index,
                                const OccupancyVoxel* voxel_ptr) {
    num_checked_voxels++;
    const float initial_log_odds =
        encodeIndexToLogOdds(block_index, voxel_index);
    if (initial_log_odds >= 0.f) {
      // Occupancy decay
      const float occupied_region_decay_log_odds = logOddsFromProbability(
          decay_integrator.occupied_region_decay_probability());
      if (initial_log_odds + occupied_region_decay_log_odds < 0.0f) {
        // Fully decayed (log odds would pass 0.5)
        EXPECT_EQ(voxel_ptr->log_odds, 0.0f);
      } else {
        // Decaying
        CHECK_NEAR(initial_log_odds + occupied_region_decay_log_odds,
                   voxel_ptr->log_odds, kEps);
      }
      // Free region decay
    } else {
      const float free_region_decay_log_odds = logOddsFromProbability(
          decay_integrator.free_region_decay_probability());
      if (initial_log_odds + free_region_decay_log_odds >= 0.0f) {
        // Fully decayed (log odds would pass 0.5)
        EXPECT_EQ(voxel_ptr->log_odds, 0.0f);
      } else {
        // Decaying
        CHECK_NEAR(initial_log_odds + free_region_decay_log_odds,
                   voxel_ptr->log_odds, kEps);
      }
    }
  };
  callFunctionOnAllVoxels<OccupancyVoxel>(&layer, check_decay_lambda);
  // Check that no blocks were deallocated
  CHECK_EQ(num_checked_voxels, kNumVoxels);
}

class OccupancyDecayParameterizedTestFixture
    : public ::testing::TestWithParam<float> {
  // Empty fixture. Just to expose the parameter.
};

TEST_P(OccupancyDecayParameterizedTestFixture, DecayAll) {
  // Test that a all voxels decay eventually
  constexpr int kNumBlocksToAllocate = 100;
  constexpr int kNumVoxels =
      kNumBlocksToAllocate * pow(OccupancyBlock::kVoxelsPerSide, 3);
  constexpr float kMaxLogOdds = 200.f;
  OccupancyLayer layer = initLayer(0.05f, kNumBlocksToAllocate, kMaxLogOdds);

  // Set up decay integrator
  constexpr float decay_step_size_log_odds = 1.5;
  constexpr int num_decays_until_converged =
      kMaxLogOdds / decay_step_size_log_odds + 1;
  OccupancyDecayIntegrator decay_integrator;
  decay_integrator.free_region_decay_probability(
      probabilityFromLogOdds(decay_step_size_log_odds));
  decay_integrator.occupied_region_decay_probability(
      probabilityFromLogOdds(-decay_step_size_log_odds));
  decay_integrator.deallocate_decayed_blocks(false);

  // The value to decay to
  const float decay_to_probability = GetParam();
  decay_integrator.decay_to_probability(decay_to_probability);

  // Decay everything
  for (size_t i = 0; i < num_decays_until_converged; i++) {
    decay_integrator.decay(&layer, CudaStreamOwning());
  }

  // Fully decayed value
  const float fully_decayed_value_log_odds =
      logOddsFromProbability(decay_integrator.decay_to_probability());

  // Check that all voxels are decayed
  int num_checked_voxels = 0;
  auto check_decay_lambda =
      [&num_checked_voxels, &fully_decayed_value_log_odds](
          const Index3D&, const Index3D&, const OccupancyVoxel* voxel_ptr) {
        num_checked_voxels++;
        EXPECT_EQ(voxel_ptr->log_odds, fully_decayed_value_log_odds);
      };
  callFunctionOnAllVoxels<OccupancyVoxel>(&layer, check_decay_lambda);
  // Check that no blocks were deallocated
  CHECK_EQ(num_checked_voxels, kNumVoxels);

  // Now test decay with deallocation
  decay_integrator.deallocate_decayed_blocks(true);
  const std::vector<Index3D> deallocated_blocks =
      decay_integrator.decay(&layer, CudaStreamOwning());
  int num_allocated_voxels = 0;
  auto check_deallocation = [&num_allocated_voxels](const Index3D&,
                                                    const Index3D&,
                                                    const OccupancyVoxel*) {
    num_allocated_voxels++;
  };
  callFunctionOnAllVoxels<OccupancyVoxel>(&layer, check_deallocation);
  // All blocks should be deallocated
  EXPECT_EQ(num_allocated_voxels, 0);
  EXPECT_EQ(deallocated_blocks.size(), kNumBlocksToAllocate);
}

// Run the above test with different values to decay to
INSTANTIATE_TEST_CASE_P(DecayAll, OccupancyDecayParameterizedTestFixture,
                        ::testing::Values(0.5f, 0.4f));

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
