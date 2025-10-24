/*
Copyright 2024 NVIDIA CORPORATION

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

#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/tests/integrator_utils.h"

using namespace nvblox;

DECLARE_bool(alsologtostderr);

class WorkspaceBoundsTest : public ::testing::Test {
 protected:
  WorkspaceBoundsTest()
      : camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  // Test camera
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

int checkWorkspaceIntegration(const Camera& camera,
                              const WorkspaceBoundsType bounds_type) {
  // Set up layer and integrator.
  constexpr float voxel_size_m = 0.1;
  TsdfLayer tsdf_layer(voxel_size_m, MemoryType::kUnified);
  ProjectiveTsdfIntegrator integrator;

  // Set the workspace bounds.
  Vector3f min_corner(-3.0f, -3.0f, 2.0f);
  Vector3f max_corner(3.0f, 3.0f, 4.0f);
  integrator.view_calculator().workspace_bounds_type(bounds_type);
  integrator.view_calculator().workspace_bounds_min_corner_m(min_corner);
  integrator.view_calculator().workspace_bounds_max_corner_m(max_corner);

  // Plane centered at (0,0,depth)
  const float kPlaneDistance = 5.0f;
  const primitives::Plane plane = primitives::Plane(
      Vector3f(0.0f, 0.0f, kPlaneDistance), Vector3f(0.0f, 0.0f, -1.0f));

  // Get a depth map of our view of the plane.
  const DepthImage depth_frame = test_utils::getPlaneDepthImage(plane, camera);

  // Integrate a frame
  std::vector<Index3D> updated_blocks;
  integrator.integrateFrame(
      MaskedDepthImageConstView(depth_frame, kMaskActiveEverywhere),
      Transform::Identity(), camera, &tsdf_layer, &updated_blocks);

  // Check that something actually happened
  EXPECT_GT(updated_blocks.size(), 0);

  int num_blocks = 0;
  for (const Index3D& block_idx : tsdf_layer.getAllBlockIndices()) {
    // Get each voxel and it's position
    auto block_ptr = tsdf_layer.getBlockAtIndex(block_idx);
    constexpr int kVoxelsPerSide = TsdfBlock::kVoxelsPerSide;
    bool has_voxel_inside_bounds = false;
    for (int x = 0; x < kVoxelsPerSide; x++) {
      for (int y = 0; y < kVoxelsPerSide; y++) {
        for (int z = 0; z < kVoxelsPerSide; z++) {
          Vector3f position = getPositionFromBlockIndexAndVoxelIndex(
              tsdf_layer.block_size(), block_idx, Index3D(x, y, z));
          if (bounds_type == WorkspaceBoundsType::kUnbounded) {
            has_voxel_inside_bounds = true;
          } else if (bounds_type == WorkspaceBoundsType::kHeightBounds) {
            if (position.z() >= min_corner.z() &&
                position.z() <= max_corner.z()) {
              has_voxel_inside_bounds = true;
            }
          } else if (bounds_type == WorkspaceBoundsType::kBoundingBox) {
            if ((position.array() >= min_corner.array()).all() &&
                (position.array() <= max_corner.array()).all()) {
              has_voxel_inside_bounds = true;
            }
          }
        }
      }
    }
    // Check that all allocated blocks are partly of fully inside the bounds.
    CHECK(has_voxel_inside_bounds);
    num_blocks++;
  }
  LOG(INFO) << "Number of blocks for " << bounds_type
            << " workspace: " << num_blocks;
  return num_blocks;
}

TEST_F(WorkspaceBoundsTest, CheckAllocatedBlocks) {
  int num_blocks_unbounded =
      checkWorkspaceIntegration(camera_, WorkspaceBoundsType::kUnbounded);
  int num_blocks_height_bounded =
      checkWorkspaceIntegration(camera_, WorkspaceBoundsType::kHeightBounds);
  int num_blocks_bounding_box =
      checkWorkspaceIntegration(camera_, WorkspaceBoundsType::kBoundingBox);

  // Check that more restrictive bounds have less blocks.
  EXPECT_GT(num_blocks_bounding_box, 0);
  EXPECT_GT(num_blocks_height_bounded, num_blocks_bounding_box);
  EXPECT_GT(num_blocks_unbounded, num_blocks_bounding_box);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
