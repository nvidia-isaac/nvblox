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
#include "nvblox/integrators/tsdf_decay_integrator.h"

#include "nvblox/primitives/scene.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

class TsdfDecayIntegratorTest : public ::testing::Test {
 protected:
  static constexpr float kVoxelSizeM{0.2};
  static constexpr float kTruncationDistanceVox{2};
  static constexpr float kTruncationDistanceMeters{kTruncationDistanceVox *
                                                   kVoxelSizeM};

  void SetUp() override {
    // Generate a TSDF layer
    primitives::Scene scene = test_utils::getSphereInBox();
    scene.generateLayerFromScene(kTruncationDistanceMeters, &layer_);
    EXPECT_GT(layer_.numAllocatedBlocks(), 0);
  }

  TsdfLayer layer_{kVoxelSizeM, MemoryType::kHost};
};

// Test behaviour of the corner case of empty layer
TEST(TsdfDecayIntegrator, EmptyLayer) {
  constexpr float KVoxelSize = 0.05;
  TsdfLayer layer(KVoxelSize, MemoryType::kHost);

  TsdfDecayIntegrator decay_integrator;
  decay_integrator.decay(&layer, CudaStreamOwning());

  EXPECT_EQ(layer.numAllocatedBlocks(), 0);
}

// Test that a single decay does what we would expect.
TEST_F(TsdfDecayIntegratorTest, SingleDecay) {
  std::vector<TsdfBlock*> block_ptrs = layer_.getAllBlockPointers();

  constexpr float kDecayFactor{0.75};

  // Create a decayed copy of the tsdf layer
  TsdfLayer layer_decayed(kVoxelSizeM, MemoryType::kHost);
  layer_decayed.copyFrom(layer_);

  TsdfDecayIntegrator decay_integrator;
  decay_integrator.deallocate_decayed_blocks(false);
  decay_integrator.decay_factor(kDecayFactor);
  decay_integrator.decay(&layer_decayed, CudaStreamOwning());

  // Check that weight decay is as expected
  auto check_weight_decay = [&layer_decayed](const Index3D& block_index,
                                             const Index3D& voxel_index,
                                             const TsdfVoxel* voxel_ptr) {
    const float original_weight = voxel_ptr->weight;
    const float decayed_weight =
        layer_decayed.getBlockAtIndex(block_index)
            ->voxels[voxel_index(0)][voxel_index(1)][voxel_index(2)]
            .weight;
    EXPECT_NEAR(original_weight * kDecayFactor, decayed_weight, 1.0E-6);
  };

  callFunctionOnAllVoxels<TsdfVoxel>(&layer_, check_weight_decay);
}

// Test that a single decay does what we would expect.
TEST_F(TsdfDecayIntegratorTest, SingleDecayWithExclusionList) {
  std::vector<TsdfBlock*> block_ptrs = layer_.getAllBlockPointers();

  constexpr float kDecayFactor{0.75};

  // Create a decayed copy of the tsdf layer
  TsdfLayer layer_decayed(kVoxelSizeM, MemoryType::kHost);
  layer_decayed.copyFrom(layer_);

  TsdfDecayIntegrator decay_integrator;
  decay_integrator.deallocate_decayed_blocks(false);
  decay_integrator.decay_factor(kDecayFactor);

  // Exclude even indices
  std::vector<Index3D> excluded_indices =
      layer_.getBlockIndicesIf([](const Index3D& index) {
        return static_cast<int>((index[0]) % 2) == 0 ||
               static_cast<int>((index[1]) % 2) == 0 ||
               static_cast<int>((index[2]) % 2) == 0;
      });
  ASSERT_TRUE(excluded_indices.size() > 0);
  decay_integrator.decay(&layer_decayed, excluded_indices, {}, {},
                         CudaStreamOwning());

  // Check that weight has not changed for blocks in exclusion list
  for (const auto& block_index : excluded_indices) {
    callFunctionOnAllVoxels<TsdfVoxel>(
        layer_.getBlockAtIndex(block_index).get(),
        [&block_index, &layer_decayed](const Index3D& voxel_index,
                                       const TsdfVoxel* voxel_ptr) {
          const float decayed_weight =
              layer_decayed.getBlockAtIndex(block_index)
                  ->voxels[voxel_index(0)][voxel_index(1)][voxel_index(2)]
                  .weight;

          EXPECT_NEAR(voxel_ptr->weight, decayed_weight, 1.0E-6);
        });
  }
}

TEST_F(TsdfDecayIntegratorTest, SingleDecayWithRadialExclusion) {
  std::vector<TsdfBlock*> block_ptrs = layer_.getAllBlockPointers();

  constexpr float kDecayFactor{0.75};

  // Create a decayed copy of the tsdf layer
  TsdfLayer layer_decayed(kVoxelSizeM, MemoryType::kHost);
  layer_decayed.copyFrom(layer_);

  TsdfDecayIntegrator decay_integrator;
  decay_integrator.deallocate_decayed_blocks(false);
  decay_integrator.decay_factor(kDecayFactor);

  constexpr float kExclusionRadiusSq = 0.025;
  const float kExclusionRadius = std::sqrt(kExclusionRadiusSq);
  const Vector3f exclusion_center = {1., 1., 1.};
  decay_integrator.decay(&layer_decayed, {}, exclusion_center, kExclusionRadius,
                         CudaStreamOwning());

  // Check that weight has not changed for blocks inside radius
  auto check_weight_decay = [&layer_decayed, &exclusion_center](
                                const Index3D& block_index,
                                const Index3D& voxel_index,
                                const TsdfVoxel* voxel_ptr) {
    const float original_weight = voxel_ptr->weight;
    const float decayed_weight =
        layer_decayed.getBlockAtIndex(block_index)
            ->voxels[voxel_index(0)][voxel_index(1)][voxel_index(2)]
            .weight;
    if ((getPositionFromBlockIndex(layer_decayed.block_size(), block_index) -
         exclusion_center)
            .squaredNorm() < kExclusionRadiusSq) {
      CHECK_EQ(original_weight, decayed_weight);
    } else {
      CHECK_NEAR(original_weight * kDecayFactor, decayed_weight, 1.0E-6);
    }
  };

  callFunctionOnAllVoxels<TsdfVoxel>(&layer_, check_weight_decay);
}

// Test that all blocks eventually decay
TEST_F(TsdfDecayIntegratorTest, DecayUntilRemoved) {
  TsdfDecayIntegrator decay_integrator;
  constexpr size_t kMaxNumIterations{1000};
  size_t num_iterations = 0;
  while (layer_.numAllocatedBlocks() > 0 &&
         num_iterations < kMaxNumIterations) {
    decay_integrator.decay(&layer_, CudaStreamOwning());
    ++num_iterations;
  }

  EXPECT_GT(num_iterations, 0);
  EXPECT_EQ(layer_.numAllocatedBlocks(), 0);
}

bool isAtLeastOneVoxelAboveWeight(const TsdfLayer& tsdf_layer,
                                  const float min_weight) {
  bool at_least_one_above = false;
  callFunctionOnAllVoxels<TsdfVoxel>(
      tsdf_layer, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
        constexpr float kEps = 1e-6;
        if (voxel->weight > (min_weight + kEps)) {
          at_least_one_above = true;
        }
      });
  return at_least_one_above;
}

std::pair<int, int> countObservedVoxels(const TsdfLayer& tsdf_layer) {
  int observed_count = 0;
  int unobserved_count = 0;
  callFunctionOnAllVoxels<TsdfVoxel>(
      tsdf_layer, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
        constexpr float kEps = 1e-6;
        if (voxel->weight > kEps) {
          ++observed_count;
        } else {
          ++unobserved_count;
        }
      });
  return {observed_count, unobserved_count};
}

TEST_F(TsdfDecayIntegratorTest, TsdfDecayToFree) {
  TsdfDecayIntegrator decay_integrator;
  constexpr size_t kMaxNumIterations{1000};
  size_t num_iterations = 0;

  // Check number of (un)observed voxels before decay
  const auto [observed_count_before, unobserved_count_before] =
      countObservedVoxels(layer_);

  // Settings under-test
  decay_integrator.set_free_distance_on_decayed(true);
  decay_integrator.deallocate_decayed_blocks(false);

  const float weight_at_decayed = decay_integrator.decayed_weight_threshold();
  const float distance_at_decayed_when_decay_to_free =
      decay_integrator.free_distance_vox() * layer_.voxel_size();

  EXPECT_TRUE(isAtLeastOneVoxelAboveWeight(
      layer_, decay_integrator.decayed_weight_threshold()));

  while (isAtLeastOneVoxelAboveWeight(layer_, weight_at_decayed) &&
         num_iterations < kMaxNumIterations) {
    decay_integrator.decay(&layer_, CudaStreamOwning());
    ++num_iterations;
  }
  EXPECT_GT(layer_.numAllocatedBlocks(), 0);

  // All voxels/blocks are fully decayed: Check
  // - Weight fully decayed
  // - Distance is set to free
  callFunctionOnAllVoxels<TsdfVoxel>(
      layer_, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
        // Only check observed voxels
        if (voxel->weight > 0.f) {
          constexpr float kEps = 1e-6;
          EXPECT_NEAR(voxel->weight, weight_at_decayed, kEps);
          EXPECT_NEAR(voxel->distance, distance_at_decayed_when_decay_to_free,
                      kEps);
        }
      });

  // Need to check that unobserved voxels are still unobserved
  const auto [observed_count_after, unobserved_count_after] =
      countObservedVoxels(layer_);

  EXPECT_EQ(observed_count_before, observed_count_after);
  EXPECT_EQ(unobserved_count_before, unobserved_count_after);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
