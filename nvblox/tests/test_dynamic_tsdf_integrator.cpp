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

#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/map/accessors.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/sensor_fixture.h"

using namespace nvblox;

class DynamicTsdfIntegratorTest : public test_utils::SensorFixture<Camera> {
 protected:
  DynamicTsdfIntegratorTest()
      : layer_(kVoxelSizeM, MemoryType::kUnified),
        reference_layer_(kVoxelSizeM, MemoryType::kUnified) {}

  DepthImage makePlaneDepthImage(float depth_m) {
    primitives::Plane plane(Vector3f(0.f, 0.f, depth_m),
                            Vector3f(0.f, 0.f, -1.f));
    return test_utils::getPlaneDepthImage(plane, sensor());
  }

  static constexpr float kVoxelSizeM = 0.05f;
  static constexpr float kTruncationDistanceVox = 4.0f;
  static constexpr float kMaxIntegrationDistanceM = 10.0f;
  static constexpr float kDiscrepancyThresholdM = 0.03f;
  static constexpr float kDynamicDiscrepancyMinWeight = 2.0f;
  static constexpr int kNumEstablishFrames = 5;

  TsdfLayer layer_;
  TsdfLayer reference_layer_;
};

TEST_F(DynamicTsdfIntegratorTest, StaticSceneMatchesStandardTsdf) {
  ProjectiveTsdfIntegrator standard_integrator;
  standard_integrator.truncation_distance_vox(kTruncationDistanceVox);
  standard_integrator.max_integration_distance_m(kMaxIntegrationDistanceM);
  standard_integrator.weighting_function_type(
      WeightingFunctionType::kConstantWeight);

  ProjectiveTsdfIntegrator dynamic_integrator;
  dynamic_integrator.truncation_distance_vox(kTruncationDistanceVox);
  dynamic_integrator.max_integration_distance_m(kMaxIntegrationDistanceM);
  dynamic_integrator.weighting_function_type(
      WeightingFunctionType::kConstantWeight);
  dynamic_integrator.dynamic_discrepancy_threshold_m(kDiscrepancyThresholdM);
  dynamic_integrator.dynamic_discrepancy_min_weight(
      kDynamicDiscrepancyMinWeight);

  const Transform T_L_C = Transform::Identity();
  DepthImage depth_frame = makePlaneDepthImage(5.0f);

  for (int i = 0; i < kNumEstablishFrames; ++i) {
    standard_integrator.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_L_C,
        sensor(), &reference_layer_);
    dynamic_integrator.integrateFrame(
        MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere), T_L_C,
        sensor(), &layer_);
  }

  const float truncation_m = kTruncationDistanceVox * kVoxelSizeM;
  int voxels_checked = 0;
  int voxels_mismatched = 0;
  callFunctionOnAllVoxels<TsdfVoxel>(
      layer_, [&](const Index3D& block_idx, const Index3D& voxel_idx,
                  const TsdfVoxel* voxel) {
        if (voxel->weight <= 0.0f) return;
        if (std::abs(voxel->distance) > truncation_m * 0.8f) return;
        const auto block = reference_layer_.getBlockAtIndex(block_idx);
        if (!block) return;
        const TsdfVoxel* ref_voxel =
            &block->voxels[voxel_idx.x()][voxel_idx.y()][voxel_idx.z()];
        if (ref_voxel->weight <= 0.0f) return;
        if (std::abs(ref_voxel->distance) > truncation_m * 0.8f) return;
        if (std::abs(voxel->distance - ref_voxel->distance) > 1e-4f ||
            std::abs(voxel->weight - ref_voxel->weight) > 1e-4f) {
          ++voxels_mismatched;
        }
        ++voxels_checked;
      });

  ASSERT_GT(voxels_checked, 0)
      << "No voxels were compared -- integration may have failed";
  EXPECT_EQ(voxels_mismatched, 0)
      << voxels_mismatched << " of " << voxels_checked
      << " near-surface voxels differ between standard and dynamic TSDF";
  LOG(INFO) << "Static equivalence: compared " << voxels_checked << " voxels, "
            << voxels_mismatched << " mismatched";
}

TEST_F(DynamicTsdfIntegratorTest, LargeShiftInvalidatesVoxels) {
  ProjectiveTsdfIntegrator integrator;
  integrator.truncation_distance_vox(kTruncationDistanceVox);
  integrator.max_integration_distance_m(kMaxIntegrationDistanceM);
  integrator.weighting_function_type(WeightingFunctionType::kConstantWeight);
  integrator.dynamic_discrepancy_threshold_m(kDiscrepancyThresholdM);
  integrator.dynamic_discrepancy_min_weight(kDynamicDiscrepancyMinWeight);

  const Transform T_L_C = Transform::Identity();
  DepthImage depth_original = makePlaneDepthImage(5.0f);

  for (int i = 0; i < kNumEstablishFrames; ++i) {
    integrator.integrateFrame(
        MaskedDepthImageConstView(depth_original, kMaskActiveEverywhere), T_L_C,
        sensor(), &layer_);
  }

  float max_weight_before = 0.0f;
  callFunctionOnAllVoxels<TsdfVoxel>(
      layer_, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
        max_weight_before = std::max(max_weight_before, voxel->weight);
      });
  EXPECT_GE(max_weight_before, kDynamicDiscrepancyMinWeight);

  DepthImage depth_shifted = makePlaneDepthImage(5.05f);
  integrator.integrateFrame(
      MaskedDepthImageConstView(depth_shifted, kMaskActiveEverywhere), T_L_C,
      sensor(), &layer_);

  int near_surface_voxels = 0;
  int low_weight_voxels = 0;
  const float truncation_m = kTruncationDistanceVox * kVoxelSizeM;
  callFunctionOnAllVoxels<TsdfVoxel>(
      layer_, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
        if (voxel->weight > 0.0f && std::abs(voxel->distance) < truncation_m) {
          ++near_surface_voxels;
          if (voxel->weight <= kDynamicDiscrepancyMinWeight) {
            ++low_weight_voxels;
          }
        }
      });

  ASSERT_GT(near_surface_voxels, 0);
  const float invalidated_fraction =
      static_cast<float>(low_weight_voxels) / near_surface_voxels;
  EXPECT_GT(invalidated_fraction, 0.5f)
      << "Expected majority of near-surface voxels to have been invalidated "
         "after a 5cm shift (threshold=3cm). Got "
      << low_weight_voxels << "/" << near_surface_voxels;
  LOG(INFO) << "Large shift: " << low_weight_voxels << "/"
            << near_surface_voxels << " voxels invalidated ("
            << invalidated_fraction * 100.0f << "%)";
}

TEST_F(DynamicTsdfIntegratorTest, SmallShiftDoesNotInvalidate) {
  ProjectiveTsdfIntegrator integrator;
  integrator.truncation_distance_vox(kTruncationDistanceVox);
  integrator.max_integration_distance_m(kMaxIntegrationDistanceM);
  integrator.weighting_function_type(WeightingFunctionType::kConstantWeight);
  integrator.dynamic_discrepancy_threshold_m(kDiscrepancyThresholdM);
  integrator.dynamic_discrepancy_min_weight(kDynamicDiscrepancyMinWeight);

  const Transform T_L_C = Transform::Identity();
  DepthImage depth_original = makePlaneDepthImage(5.0f);

  for (int i = 0; i < kNumEstablishFrames; ++i) {
    integrator.integrateFrame(
        MaskedDepthImageConstView(depth_original, kMaskActiveEverywhere), T_L_C,
        sensor(), &layer_);
  }

  DepthImage depth_shifted = makePlaneDepthImage(5.02f);
  integrator.integrateFrame(
      MaskedDepthImageConstView(depth_shifted, kMaskActiveEverywhere), T_L_C,
      sensor(), &layer_);

  int near_surface_voxels = 0;
  int high_weight_voxels = 0;
  const float truncation_m = kTruncationDistanceVox * kVoxelSizeM;
  callFunctionOnAllVoxels<TsdfVoxel>(
      layer_, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
        if (voxel->weight > 0.0f && std::abs(voxel->distance) < truncation_m) {
          ++near_surface_voxels;
          if (voxel->weight > kDynamicDiscrepancyMinWeight) {
            ++high_weight_voxels;
          }
        }
      });

  ASSERT_GT(near_surface_voxels, 0);
  const float retained_fraction =
      static_cast<float>(high_weight_voxels) / near_surface_voxels;
  EXPECT_GT(retained_fraction, 0.9f)
      << "Expected most near-surface voxels to retain high weight after a "
         "2cm shift (threshold=3cm). Got "
      << high_weight_voxels << "/" << near_surface_voxels;
  LOG(INFO) << "Small shift: " << high_weight_voxels << "/"
            << near_surface_voxels << " voxels retained high weight ("
            << retained_fraction * 100.0f << "%)";
}

TEST_F(DynamicTsdfIntegratorTest, OccludedVoxelsNotInvalidated) {
  ProjectiveTsdfIntegrator integrator;
  integrator.truncation_distance_vox(kTruncationDistanceVox);
  integrator.max_integration_distance_m(kMaxIntegrationDistanceM);
  integrator.weighting_function_type(WeightingFunctionType::kConstantWeight);
  integrator.dynamic_discrepancy_threshold_m(kDiscrepancyThresholdM);
  integrator.dynamic_discrepancy_min_weight(kDynamicDiscrepancyMinWeight);

  const Transform T_L_C = Transform::Identity();
  DepthImage depth_background = makePlaneDepthImage(5.0f);

  for (int i = 0; i < kNumEstablishFrames; ++i) {
    integrator.integrateFrame(
        MaskedDepthImageConstView(depth_background, kMaskActiveEverywhere),
        T_L_C, sensor(), &layer_);
  }

  int established_voxels_before = 0;
  callFunctionOnAllVoxels<TsdfVoxel>(
      layer_, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
        if (voxel->weight >= kDynamicDiscrepancyMinWeight) {
          ++established_voxels_before;
        }
      });
  ASSERT_GT(established_voxels_before, 0);

  // Introduce foreground at z=2m (far in front of background)
  DepthImage depth_foreground = makePlaneDepthImage(2.0f);
  integrator.integrateFrame(
      MaskedDepthImageConstView(depth_foreground, kMaskActiveEverywhere), T_L_C,
      sensor(), &layer_);

  // Check that background voxels were not invalidated by counting voxels
  // that still have weight >= kDynamicDiscrepancyMinWeight at the background
  // depth.
  int established_voxels_after = 0;
  callFunctionOnAllVoxels<TsdfVoxel>(
      layer_, [&](const Index3D&, const Index3D&, const TsdfVoxel* voxel) {
        if (voxel->weight >= kDynamicDiscrepancyMinWeight) {
          ++established_voxels_after;
        }
      });

  // Free-space voxels near the new foreground surface (within its truncation
  // band) will be legitimately invalidated. But the vast majority of
  // established background voxels (at z~5m) should be preserved.
  const float retained_fraction =
      static_cast<float>(established_voxels_after) / established_voxels_before;
  EXPECT_GT(retained_fraction, 0.9f)
      << "Too many background voxels were invalidated by a foreground surface. "
         "Retained "
      << established_voxels_after << "/" << established_voxels_before;
  LOG(INFO) << "Occlusion guard: established voxels before="
            << established_voxels_before
            << ", after=" << established_voxels_after << " (retained "
            << retained_fraction * 100.0f << "%)";
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  ::gflags::ParseCommandLineFlags(&argc, &argv, true);
  return RUN_ALL_TESTS();
}
