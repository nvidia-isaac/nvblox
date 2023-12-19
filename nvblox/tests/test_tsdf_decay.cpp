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
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/io/image_io.h"
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
    scene_ = test_utils::getSphereInBox();
    scene_.generateLayerFromScene(kTruncationDistanceMeters, &layer_);
    EXPECT_GT(layer_.numAllocatedBlocks(), 0);
  }

  primitives::Scene scene_;
  TsdfLayer layer_{kVoxelSizeM, MemoryType::kHost};

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_ = Camera(fu_, fv_, cu_, cv_, width_, height_);
};

// Test behaviour of the corner case of empty layer
TEST(TsdfDecayIntegrator, EmptyLayer) {
  constexpr float KVoxelSize = 0.05;
  TsdfLayer layer(KVoxelSize, MemoryType::kHost);

  TsdfDecayIntegrator decay_integrator;
  const std::vector<Index3D> dellocated_blocks =
      decay_integrator.decay(&layer, CudaStreamOwning());

  EXPECT_EQ(layer.numAllocatedBlocks(), 0);
  EXPECT_TRUE(dellocated_blocks.empty());
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

  decay_integrator.decay(
      &layer_decayed,
      DecayBlockExclusionOptions{.block_indices_to_exclude = excluded_indices},
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
  const DecayBlockExclusionOptions exclusions_options{
      .block_indices_to_exclude = {},
      .exclusion_center = exclusion_center,
      .exclusion_radius_m = kExclusionRadius};
  decay_integrator.decay(&layer_decayed, exclusions_options,
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
  const int num_allocated_blocks = layer_.numAllocatedBlocks();
  int num_dellocated_blocks = 0;
  while (layer_.numAllocatedBlocks() > 0 &&
         num_iterations < kMaxNumIterations) {
    const auto decayed_this_iteration =
        decay_integrator.decay(&layer_, CudaStreamOwning());
    num_dellocated_blocks += decayed_this_iteration.size();
    ++num_iterations;
  }

  EXPECT_GT(num_iterations, 0);
  EXPECT_EQ(layer_.numAllocatedBlocks(), 0);
  EXPECT_EQ(num_allocated_blocks, num_dellocated_blocks);
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

  int num_dellocated_blocks = 0;
  while (isAtLeastOneVoxelAboveWeight(layer_, weight_at_decayed) &&
         num_iterations < kMaxNumIterations) {
    const auto deallocated_this_iteration =
        decay_integrator.decay(&layer_, CudaStreamOwning());
    ++num_iterations;
    num_dellocated_blocks += deallocated_this_iteration.size();
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

bool isVoxelInView(const DepthImage& depth_image, const Index3D& block_idx,
                   const Index3D& voxel_idx, const float block_size_m,
                   const Camera& camera, const Transform& T_L_C,
                   const float max_depth_m, const float truncation_distance_m) {
  // Project the voxel onto the image plane, failing if the voxel is too far
  // away or outside the bounds of the image.
  const Vector3f p_L = getCenterPositionFromBlockIndexAndVoxelIndex(
      block_size_m, block_idx, voxel_idx);
  const Vector3f p_C = T_L_C.inverse() * p_L;
  if (p_C.z() > max_depth_m) {
    return false;
  }
  Vector2f u_C;
  const bool on_image = camera.project(p_C, &u_C);
  if (!on_image) {
    return false;
  }
  // Interpolate the measurement for the depth.
  float surface_depth_measured;
  if (!interpolation::interpolate2DClosest<
          float, interpolation::checkers::FloatPixelGreaterThanZero>(
          depth_image, u_C, &surface_depth_measured)) {
    return false;
  }
  // Get the projective SDF value
  const float voxel_depth_m = p_C.z();
  const float voxel_to_surface_distance =
      surface_depth_measured - voxel_depth_m;
  const bool occluded = (voxel_to_surface_distance < -truncation_distance_m);
  // The voxel is in view finally if it's not occluded
  return !occluded;
}

TEST_F(TsdfDecayIntegratorTest, TsdfDecayExcludeView) {
  // Get a depth image of the scene
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);

  // Generate a depth image of the scene.
  // NOTE(alexmillane): Looking along the xaxis.
  Transform T_L_C = Transform::Identity();
  T_L_C.prerotate(Eigen::Quaternionf(0.5, 0.5, 0.5, 0.5));
  T_L_C.pretranslate(Vector3f(-4.0f, 0.0f, 2.0f));
  // This value for max distance generates some invalid regions.
  constexpr float kMaxDist = 10.f;
  scene_.generateDepthImageFromScene(camera_, T_L_C, kMaxDist, &depth_frame);

  // Get the original weight (read it from a random voxel).
  const TsdfBlock::ConstPtr tsdf_block =
      layer_.getBlockAtIndex(Index3D(0, 0, 0));
  CHECK(tsdf_block);
  const float original_weight = tsdf_block->voxels[0][0][0].weight;
  CHECK(original_weight > 0.f);

  // Start to decay
  TsdfDecayIntegrator decay_integrator;
  const float kMaxViewDistanceM = kMaxDist;
  const std::vector<Index3D> deallocated_blocks = decay_integrator.decay(
      &layer_,
      DecayViewExclusionOptions(&depth_frame, T_L_C, camera_, kMaxViewDistanceM,
                                kTruncationDistanceMeters),
      CudaStreamOwning());
  // We expect that no blocks are deallocated after a single decay.
  EXPECT_EQ(deallocated_blocks.size(), 0);

  // Go over voxels and check that only those in view have decayed
  int num_decayed = 0;
  int num_not_decayed = 0;
  // int num_not_decayed_in_view = 0;
  int num_not_valid = 0;
  int num_in_view = 0;
  int num_not_in_view = 0;
  callFunctionOnAllVoxels<TsdfVoxel>(layer_, [&](const Index3D& block_idx,
                                                 const Index3D& voxel_idx,
                                                 const TsdfVoxel* voxel) {
    constexpr float kEps = 1e-3;
    // Some of the voxels in the scene are outside of the bounds and
    // therefore don't have a valid distance or a weight above zero. We
    // exclude these from the test.
    const bool not_in_scene = (voxel->weight < kEps);
    if (not_in_scene) {
      num_not_valid++;
      return;
    }
    // Voxel not decayed if it still has its original weight
    const bool not_decayed = (original_weight - voxel->weight) < kEps;
    // In view
    const bool is_in_view = isVoxelInView(
        depth_frame, block_idx, voxel_idx, layer_.block_size(), camera_, T_L_C,
        kMaxViewDistanceM, kTruncationDistanceMeters);
    // The check - A voxel is either:
    //   - decayed and out of view, or
    //   - not decayed and in view.
    EXPECT_TRUE((not_decayed && is_in_view) || (!not_decayed && !is_in_view));
    // Counts
    if (is_in_view) {
      ++num_in_view;
    } else {
      ++num_not_in_view;
    }
    if (not_decayed) {
      num_not_decayed++;
    } else {
      num_decayed++;
    }
  });
  EXPECT_EQ(num_not_decayed, num_in_view);
  EXPECT_EQ(num_decayed, num_not_in_view);
  EXPECT_GT(num_decayed, 0);
  EXPECT_GT(num_not_decayed, 0);

  constexpr bool kTestOutput = false;
  if (kTestOutput) {
    io::writeToPng("tsdf_decay_image.png", depth_frame);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
