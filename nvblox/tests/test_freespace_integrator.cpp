/*
Copyright 2023 NVIDIA CORPORATION

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

#include "nvblox/integrators/freespace_integrator.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/tests/integrator_utils.h"

using namespace nvblox;

class FreespaceIntegratorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    timing::Timing::Reset();
    std::srand(0);
    block_size_m_ = VoxelBlock<bool>::kVoxelsPerSide * voxel_size_m_;

    // Arbitrary camera
    constexpr float fu = 300;
    constexpr float fv = 300;
    constexpr int width = 640;
    constexpr int height = 480;
    constexpr float cu = static_cast<float>(width) / 2.0f;
    constexpr float cv = static_cast<float>(height) / 2.0f;
    camera_ = Camera(fu, fv, cu, cv, width, height);
  }

  float voxel_size_m_ = 0.05;
  float block_size_m_ = TsdfBlock::kVoxelsPerSide * voxel_size_m_;
  Camera camera_;

  primitives::Scene scene_;
};

/// Check that the freespace_voxel is correct (i.e. equal to the
/// freespace_voxel_gt).
bool checkVoxel(const FreespaceVoxel& freespace_voxel,
                const TsdfVoxel& tsdf_voxel,
                const FreespaceVoxel& freespace_voxel_gt,
                float max_tsdf_distance_to_check) {
  if (tsdf_voxel.weight <= 1e-4) {
    return false;
  }
  if (tsdf_voxel.distance >= max_tsdf_distance_to_check) {
    return false;  // We do not want to check these.
  }
  EXPECT_EQ(freespace_voxel.last_occupied_timestamp_ms,
            freespace_voxel_gt.last_occupied_timestamp_ms);
  EXPECT_EQ(freespace_voxel.consecutive_occupancy_duration_ms,
            freespace_voxel_gt.consecutive_occupancy_duration_ms);
  EXPECT_EQ(freespace_voxel.is_high_confidence_freespace,
            freespace_voxel_gt.is_high_confidence_freespace);
  return true;
}

/// Check that all voxels of the freespace layer are correct (i.e. equal to the
/// freespace_voxel_gt).
void checkVoxels(
    const FreespaceLayer& freespace_layer, const TsdfLayer& tsdf_layer,
    const std::vector<Index3D>& block_indices,
    const FreespaceVoxel& freespace_voxel_gt,
    float max_tsdf_distance_to_check = std::numeric_limits<float>::max()) {
  int num_checked_voxels = 0;
  for (const Index3D& idx : block_indices) {
    // Get block pointers and check if allocated
    EXPECT_TRUE(freespace_layer.isBlockAllocated(idx));
    const auto freespace_block = freespace_layer.getBlockAtIndex(idx);
    EXPECT_TRUE(tsdf_layer.isBlockAllocated(idx));
    const auto tsdf_block = tsdf_layer.getBlockAtIndex(idx);

    // Iterate over all voxels in the block
    constexpr int kVoxelsPerSide = VoxelBlock<FreespaceVoxel>::kVoxelsPerSide;
    Index3D voxel_index;
    for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
         voxel_index.x()++) {
      for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
           voxel_index.y()++) {
        for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
             voxel_index.z()++) {
          // Get the voxels
          FreespaceVoxel freespace_voxel =
              freespace_block
                  ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];
          TsdfVoxel tsdf_voxel =
              tsdf_block
                  ->voxels[voxel_index.x()][voxel_index.y()][voxel_index.z()];

          // Check the freespace voxel
          if (checkVoxel(freespace_voxel, tsdf_voxel, freespace_voxel_gt,
                         max_tsdf_distance_to_check)) {
            num_checked_voxels++;
          }
        }
      }
    }
  }
  CHECK_GT(num_checked_voxels, 0);
}

TEST_F(FreespaceIntegratorTest, FreespacePlane) {
  // Overview of test steps:
  // (1) create an empty scene
  // (2) update tsdf/freespace layer and check initialization
  // (3) update and check that:
  //     - consecutive_occupancy_duration_ms is increasing
  // (4) update and check that:
  //     - consecutive_occupancy_duration_ms is terminated after
  //       max_unobserved_to_keep_consecutive_occupancy_ms
  // (5) update and check that:
  //     - we change to is_high_confidence_freespace after
  //       min_duration_since_occupied_for_freespace_ms
  // (6) add a plane to the scene to make voxels occupied
  // (7) update and check that:
  //     - last_occupied_timestamp_ms is updated
  //     - is_high_confidence_freespace is still set until
  //       min_consecutive_occupancy_duration_for_reset_ms
  // (8) update and check that:
  //     - consecutive_occupancy_duration_ms is increasing
  //     - is_high_confidence_freespace is still set until
  //       min_consecutive_occupancy_duration_for_reset_ms
  // (8) update and check that:
  //     - is_high_confidence_freespace is reset after
  //       min_consecutive_occupancy_duration_for_reset_ms

  // We create a scene that is a flat plane 4 meters from the origin.
  constexpr float kPlaneDistance = 4.0f;
  scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(kPlaneDistance, 0.0, 0.0), Vector3f(-1, 0, 0)));

  // Create a pose at the origin looking forward.
  Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
  Transform T_L_C = Transform::Identity();
  T_L_C.prerotate(rotation_base);

  // Generate a depth frame
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);
  // Make sure the full plane is visible.
  const float depth_image_max_distance_m = 2.0f * kPlaneDistance;
  scene_.generateDepthImageFromScene(camera_, T_L_C, depth_image_max_distance_m,
                                     &depth_frame);

  // Layers
  FreespaceLayer freespace_layer(voxel_size_m_, MemoryType::kUnified);
  TsdfLayer tsdf_layer(voxel_size_m_, MemoryType::kUnified);

  // Setup Tsdf integrator
  ProjectiveTsdfIntegrator tsdf_integrator;
  constexpr float kTruncationDistVox = 4;
  const float truncation_distance_m = kTruncationDistVox * voxel_size_m_;
  // We choose an integration distance smaller than the distance to the plane to
  // allocate only freespace for now.
  const float max_integration_distance_m =
      kPlaneDistance - 2 * truncation_distance_m;
  tsdf_integrator.truncation_distance_vox(kTruncationDistVox);
  tsdf_integrator.max_integration_distance_m(max_integration_distance_m);

  // Setup the freespace integrator.
  FreespaceIntegrator freespace_integrator;
  const Time time_step_ms{100};
  freespace_integrator.max_tsdf_distance_for_occupancy_m(0.75 *
                                                         truncation_distance_m);
  freespace_integrator.max_unobserved_to_keep_consecutive_occupancy_ms(
      2 * time_step_ms);
  freespace_integrator.min_duration_since_occupied_for_freespace_ms(
      5 * time_step_ms);
  freespace_integrator.min_consecutive_occupancy_duration_for_reset_ms(
      10 * time_step_ms);
  // TODO(remos): Also implement a test that does neighborhood checking.
  freespace_integrator.check_neighborhood(false);

  // Integrate the depth frame and update the freespace layer.
  std::vector<Index3D> updated_blocks;
  const Time start_time_ms{42};  // almost random
  tsdf_integrator.integrateFrame(depth_frame, T_L_C, camera_, &tsdf_layer,
                                 &updated_blocks);
  freespace_integrator.updateFreespaceLayer(updated_blocks, start_time_ms,
                                            tsdf_layer, &freespace_layer);
  const std::vector<Index3D> block_indices = tsdf_layer.getAllBlockIndices();

  // Check that voxels are initialized to being occupied.
  std::cout << "Check initialization." << std::endl;
  FreespaceVoxel freespace_voxel_gt;
  freespace_voxel_gt.last_occupied_timestamp_ms = start_time_ms;
  freespace_voxel_gt.consecutive_occupancy_duration_ms = Time(0);
  freespace_voxel_gt.is_high_confidence_freespace = false;
  checkVoxels(freespace_layer, tsdf_layer, block_indices, freespace_voxel_gt);

  // As the voxel was initialized to occupied, we now check that consecutive
  // occupancy starts increasing.
  std::cout << "Check consecutive occupancy is increasing." << std::endl;
  Time current_time_ms = start_time_ms + time_step_ms;
  freespace_integrator.updateFreespaceLayer(updated_blocks, current_time_ms,
                                            tsdf_layer, &freespace_layer);
  freespace_voxel_gt.consecutive_occupancy_duration_ms =
      current_time_ms - start_time_ms;
  checkVoxels(freespace_layer, tsdf_layer, block_indices, freespace_voxel_gt);

  // Check that consecutive occupancy did terminate now because we surpassed:
  // max_unobserved_to_keep_consecutive_occupancy_ms = 2 * time_step_ms
  std::cout << "Check consecutive occupancy is terminated." << std::endl;
  current_time_ms = start_time_ms + 3 * time_step_ms;
  freespace_integrator.updateFreespaceLayer(updated_blocks, current_time_ms,
                                            tsdf_layer, &freespace_layer);
  freespace_voxel_gt.consecutive_occupancy_duration_ms = Time(0);
  checkVoxels(freespace_layer, tsdf_layer, block_indices, freespace_voxel_gt);

  // Check that we change to high confidence freespace because we surpassed:
  // min_duration_since_occupied_for_freespace_ms = 5 * time_step_ms
  std::cout << "Check change to high confidence freespace." << std::endl;
  current_time_ms = start_time_ms + 5 * time_step_ms;
  freespace_integrator.updateFreespaceLayer(updated_blocks, current_time_ms,
                                            tsdf_layer, &freespace_layer);
  freespace_voxel_gt.is_high_confidence_freespace = true;
  checkVoxels(freespace_layer, tsdf_layer, block_indices, freespace_voxel_gt);

  // Add a plane in view to test switching back from high confidence freespace
  // to occupied space.
  scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(max_integration_distance_m, 0.0, 0.0), Vector3f(-1, 0, 0)));
  scene_.generateDepthImageFromScene(camera_, T_L_C, depth_image_max_distance_m,
                                     &depth_frame);

  // Update tsdf and freespace layers.
  Time plane_added_time_ms = start_time_ms + 10 * time_step_ms;
  tsdf_integrator.integrateFrame(depth_frame, T_L_C, camera_, &tsdf_layer,
                                 &updated_blocks);
  freespace_integrator.updateFreespaceLayer(updated_blocks, plane_added_time_ms,
                                            tsdf_layer, &freespace_layer);

  // Check that we updated the last occupied field but not did not abandon the
  // high confidence freespace flag yet.
  std::cout
      << "Check last occupied updated but keeping high confidence freespace."
      << std::endl;
  freespace_voxel_gt.last_occupied_timestamp_ms = plane_added_time_ms;
  freespace_voxel_gt.consecutive_occupancy_duration_ms = Time(0);
  freespace_voxel_gt.is_high_confidence_freespace = true;
  // We only check the voxels that should switch back to occupied.
  const float max_tsdf_distance_to_check =
      freespace_integrator.max_tsdf_distance_for_occupancy_m();
  checkVoxels(freespace_layer, tsdf_layer, block_indices, freespace_voxel_gt,
              max_tsdf_distance_to_check);

  std::cout << "Check increasing consecutive occupancy but keeping high "
               "confidence freespace."
            << std::endl;
  // We have to integrate every second time step to not loose consecutive
  // occupancy.
  freespace_integrator.updateFreespaceLayer(
      updated_blocks, plane_added_time_ms + 2 * time_step_ms, tsdf_layer,
      &freespace_layer);
  freespace_integrator.updateFreespaceLayer(
      updated_blocks, plane_added_time_ms + 4 * time_step_ms, tsdf_layer,
      &freespace_layer);
  freespace_integrator.updateFreespaceLayer(
      updated_blocks, plane_added_time_ms + 6 * time_step_ms, tsdf_layer,
      &freespace_layer);
  freespace_integrator.updateFreespaceLayer(
      updated_blocks, plane_added_time_ms + 8 * time_step_ms, tsdf_layer,
      &freespace_layer);
  freespace_voxel_gt.last_occupied_timestamp_ms =
      plane_added_time_ms + 8 * time_step_ms;
  freespace_voxel_gt.consecutive_occupancy_duration_ms = 8 * time_step_ms;
  freespace_voxel_gt.is_high_confidence_freespace = true;
  checkVoxels(freespace_layer, tsdf_layer, block_indices, freespace_voxel_gt,
              max_tsdf_distance_to_check);

  // Check that we reset to occupied because we surpassed:
  // min_consecutive_occupancy_duration_for_reset_ms = 10 *
  // time_step_ms
  std::cout << "Check that we reset to occupied." << std::endl;
  freespace_integrator.updateFreespaceLayer(
      updated_blocks, plane_added_time_ms + 10 * time_step_ms, tsdf_layer,
      &freespace_layer);
  freespace_voxel_gt.last_occupied_timestamp_ms =
      plane_added_time_ms + 10 * time_step_ms;
  freespace_voxel_gt.consecutive_occupancy_duration_ms = 10 * time_step_ms;
  freespace_voxel_gt.is_high_confidence_freespace = false;
  checkVoxels(freespace_layer, tsdf_layer, block_indices, freespace_voxel_gt,
              max_tsdf_distance_to_check);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
