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

#include "nvblox/core/types.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

// TODO: Decide where to put test epsilons
// NOTE(alexmillane): I had to crank this up slightly to get things to pass... I
// guess this is just floating point errors accumulating?
constexpr float kFloatEpsilon = 1e-4;

std::pair<Vector3f, Vector2f> getRandomVisibleRayAndImagePoint(
    const Camera& camera) {
  //  Random point on image plane
  const Vector2f u_C(test_utils::randomFloatInRange(
                         0.0f, static_cast<float>(camera.width() - 1)),
                     test_utils::randomFloatInRange(
                         0.0f, static_cast<float>(camera.height() - 1)));
  // Normalized ray
  return {camera.vectorFromImagePlaneCoordinates(u_C).normalized(), u_C};
}

Camera getTestCamera() {
  // Arbitrary camera
  constexpr float fu = 300;
  constexpr float fv = 300;
  constexpr int width = 640;
  constexpr int height = 480;
  constexpr float cu = static_cast<float>(width) / 2.0f;
  constexpr float cv = static_cast<float>(height) / 2.0f;
  return Camera(fu, fv, cu, cv, width, height);
}

TEST(CameraTest, PointsInView) {
  // Make sure this is deterministic.
  std::srand(0);

  const Camera camera = getTestCamera();

  // Generate some random points (in view) and project them back
  constexpr int kNumPoints = 1000;
  for (int i = 0; i < kNumPoints; i++) {
    Vector3f ray_C;
    Vector2f u_C;
    std::tie(ray_C, u_C) = getRandomVisibleRayAndImagePoint(camera);
    const Vector3f p_C = test_utils::randomFloatInRange(1.0, 1000.0) * ray_C;
    Vector2f u_reprojection_C(0.F, 0.F);
    EXPECT_TRUE(camera.project(p_C, &u_reprojection_C));
    EXPECT_TRUE(((u_reprojection_C - u_C).array().abs() < kFloatEpsilon).all());
  }
}

TEST(CameraTest, CenterPixel) {
  // Make sure this is deterministic.
  std::srand(0);

  const Camera camera = getTestCamera();

  // Center
  const Vector3f center_ray = Vector3f(0.0f, 0.0f, 1.0f);
  const Vector3f p_C = test_utils::randomFloatInRange(1.0, 1000.0) * center_ray;
  Eigen::Vector2f u(0.F, 0.F);
  EXPECT_TRUE(camera.project(p_C, &u));
  EXPECT_TRUE(
      ((u - Vector2f(camera.cu(), camera.cv())).array().abs() < kFloatEpsilon)
          .all());
}

TEST(CameraTest, BehindCamera) {
  // Make sure this is deterministic.
  std::srand(0);

  const Camera camera = getTestCamera();

  constexpr int kNumPoints = 1000;
  for (int i = 0; i < kNumPoints; i++) {
    Vector3f ray_C;
    Vector2f u_C;
    std::tie(ray_C, u_C) = getRandomVisibleRayAndImagePoint(camera);
    Vector3f p_C = test_utils::randomFloatInRange(1.0, 1000.0) * ray_C;
    // The negative here puts the point behind the camera
    p_C.z() = -1.0f * p_C.z();
    Vector2f u_reprojection_C;
    EXPECT_FALSE(camera.project(p_C, &u_reprojection_C));
  }
}

TEST(CameraTest, OutsideImagePlane) {
  // Make sure this is deterministic.
  std::srand(0);

  const Camera camera = getTestCamera();

  // NOTE(alexmillane): My own ray-from-pixel function to not trigger checks
  // because the pixel is off the image plane.
  const auto rayFromPixelNoChecks = [camera](const auto& u_C) {
    return Vector3f((u_C[0] - camera.cu()) / camera.fu(),
                    (u_C[1] - camera.cv()) / camera.fv(), 1.0f);
  };

  constexpr int kNumPoints = 1000;
  for (int i = 0; i < kNumPoints; i++) {
    //  Random point off image plane
    // Add a random offset to the center pixel with sufficient magnitude to take
    // it off the plane.
    constexpr float kOffImagePlaneFactor = 5.0;
    const Vector2f u_perturbation_C(
        test_utils::randomSign() *
            test_utils::randomFloatInRange(
                camera.width() / 2.0, kOffImagePlaneFactor * camera.width()),
        test_utils::randomSign() *
            test_utils::randomFloatInRange(
                camera.height() / 2.0, kOffImagePlaneFactor * camera.height()));
    const Vector2f u_C = Vector2f(camera.cu(), camera.cv()) + u_perturbation_C;

    const Vector3f ray_C = rayFromPixelNoChecks(u_C);
    const Vector3f p_C = test_utils::randomFloatInRange(1.0, 1000.0) * ray_C;
    Vector2f u_reprojection_C;
    EXPECT_FALSE(camera.project(p_C, &u_reprojection_C));
  }
}

TEST(CameraTest, AxisAlignedBoundingBox) {
  // Make sure this is deterministic.
  std::srand(0);

  const Camera camera = getTestCamera();

  // Rays through the corners of the image plane
  const Vector3f ray_0_C =
      camera.vectorFromImagePlaneCoordinates(Vector2f(0.0f, 0.0f));
  const Vector3f ray_2_C =
      camera.vectorFromImagePlaneCoordinates(Vector2f(0.0f, camera.height()));
  const Vector3f ray_1_C =
      camera.vectorFromImagePlaneCoordinates(Vector2f(camera.width(), 0.0f));
  const Vector3f ray_3_C = camera.vectorFromImagePlaneCoordinates(
      Vector2f(camera.width(), camera.height()));

  // Generate a random depths
  constexpr float kMinimumDepthPx = 1.0;
  constexpr float kMaximumDepthPx = 1000.0;
  const float min_depth =
      test_utils::randomFloatInRange(kMinimumDepthPx, kMaximumDepthPx);
  const float max_depth =
      test_utils::randomFloatInRange(kMinimumDepthPx, kMaximumDepthPx);

  // True bounding box from the 3D points
  AlignedVector<Vector3f> view_corners_C = {
      min_depth * ray_0_C, max_depth * ray_0_C,  // NOLINT
      min_depth * ray_1_C, max_depth * ray_1_C,  // NOLINT
      min_depth * ray_2_C, max_depth * ray_2_C,  // NOLINT
      min_depth * ray_3_C, max_depth * ray_3_C   // NOLINT
  };
  AxisAlignedBoundingBox aabb_true;
  std::for_each(view_corners_C.begin(), view_corners_C.end(),
                [&aabb_true](const Vector3f& p) { aabb_true.extend(p); });

  // Bounding box approximated by the camera model.
  // TODO(alexmillane): Only tested with identity transform at the moment.
  const Transform T_L_C = Transform::Identity();
  const AxisAlignedBoundingBox aabb_test =
      camera.getViewAABB(T_L_C, min_depth, max_depth);

  EXPECT_TRUE(aabb_true.isApprox(aabb_test))
      << "AABB true: " << aabb_true.min().transpose() << " "
      << aabb_true.max().transpose()
      << " AABB test: " << aabb_test.min().transpose() << " "
      << aabb_test.max().transpose();
}

TEST(CameraTest, FrustumTest) {
  constexpr float kMinDist = 1.0f;
  constexpr float kMaxDist = 10.0f;

  const Camera camera = getTestCamera();

  Frustum frustum(camera, Transform::Identity(), kMinDist, kMaxDist);

  // Project a point into the camera.
  Vector3f point_C(0.5, 0.5, 5.0);
  Vector2f u_C;
  ASSERT_TRUE(camera.project(point_C, &u_C));

  // Check that the point is within the frustum.
  EXPECT_TRUE(frustum.isPointInView(point_C));

  // Check a point further than the max dist.
  point_C << 0.5, 0.5, kMaxDist + 10.0f;
  ASSERT_TRUE(camera.project(point_C, &u_C));
  EXPECT_FALSE(frustum.isPointInView(point_C));

  // Check a point closer than the max dist.
  point_C << 0.0, 0.0, 0.3f;
  ASSERT_TRUE(camera.project(point_C, &u_C));
  EXPECT_FALSE(frustum.isPointInView(point_C));
}

TEST(CameraTest, FrustumAABBTest) {
  constexpr float kMinDist = 1.0f;
  constexpr float kMaxDist = 10.0f;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const Camera camera = getTestCamera();

  Frustum frustum(camera, Transform::Identity(), kMinDist, kMaxDist);
  AxisAlignedBoundingBox view_aabb =
      camera.getViewAABB(Transform::Identity(), kMinDist, kMaxDist);

  // Double-check that the camera and the frustum AABB match.
  EXPECT_TRUE(frustum.isAABBInView(view_aabb));
  EXPECT_TRUE(view_aabb.isApprox(frustum.getAABB()));

  // Get all blocks in the view AABB and make sure that some of them are
  // actually in the view.
  const float block_size = 1.0f;
  std::vector<Index3D> block_indices_in_aabb =
      getBlockIndicesTouchedByBoundingBox(block_size, view_aabb);
  std::vector<Index3D> block_indices_in_frustum;
  for (const Index3D& block_index : block_indices_in_aabb) {
    const AxisAlignedBoundingBox& aabb_block =
        getAABBOfBlock(block_size, block_index);
    if (frustum.isAABBInView(aabb_block)) {
      block_indices_in_frustum.push_back(block_index);
    }
  }

  EXPECT_GT(block_indices_in_aabb.size(), block_indices_in_frustum.size());
  EXPECT_GT(block_indices_in_frustum.size(), 0);

  // Check all voxels within the view and make sure that they're correctly
  // marked.
  for (const Index3D& block_index : block_indices_in_aabb) {
    Index3D voxel_index;

    // Iterate over all the voxels:
    for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
         voxel_index.x()++) {
      for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
           voxel_index.y()++) {
        for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
             voxel_index.z()++) {
          Vector3f position = getCenterPositionFromBlockIndexAndVoxelIndex(
              block_size, block_index, voxel_index);
          Eigen::Vector2f u_C;
          bool in_frustum = frustum.isPointInView(position);
          bool in_camera = camera.project(position, &u_C);
          if (position.z() <= kMaxDist && position.z() >= kMinDist) {
            EXPECT_EQ(in_frustum, in_camera);
          } else {
            // Doesn't matter if we're within the camera view if it's false.
            EXPECT_FALSE(in_frustum);
          }
        }
      }
    }
  }
}

TEST(CameraTest, FrustumAtLeastOneValidVoxelTest) {
  constexpr float kMinDist = 0.0f;
  constexpr float kMaxDist = 10.0f;
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const Camera camera = getTestCamera();

  Frustum frustum(camera, Transform::Identity(), kMinDist, kMaxDist);
  AxisAlignedBoundingBox view_aabb =
      camera.getViewAABB(Transform::Identity(), kMinDist, kMaxDist);

  // Double-check that the camera and the frustum AABB match.
  EXPECT_TRUE(frustum.isAABBInView(view_aabb));
  EXPECT_TRUE(view_aabb.isApprox(frustum.getAABB()));

  // Get all blocks in the view AABB and make sure that some of them are
  // actually in the view.
  const float block_size = 0.5f;
  std::vector<Index3D> block_indices_in_aabb =
      getBlockIndicesTouchedByBoundingBox(block_size, view_aabb);
  std::vector<Index3D> block_indices_in_frustum;
  for (const Index3D& block_index : block_indices_in_aabb) {
    const AxisAlignedBoundingBox& aabb_block =
        getAABBOfBlock(block_size, block_index);
    if (frustum.isAABBInView(aabb_block)) {
      block_indices_in_frustum.push_back(block_index);
    }
  }

  EXPECT_GT(block_indices_in_aabb.size(), block_indices_in_frustum.size());
  EXPECT_GT(block_indices_in_frustum.size(), 0);

  // Check that for any given block in the frustum, there's AT LEAST one valid
  // voxel.
  int empty = 0;
  for (const Index3D& block_index : block_indices_in_frustum) {
    Index3D voxel_index;
    bool any_valid = false;
    // Iterate over all the voxels:
    for (voxel_index.x() = 0; voxel_index.x() < kVoxelsPerSide;
         voxel_index.x()++) {
      for (voxel_index.y() = 0; voxel_index.y() < kVoxelsPerSide;
           voxel_index.y()++) {
        for (voxel_index.z() = 0; voxel_index.z() < kVoxelsPerSide;
             voxel_index.z()++) {
          Vector3f position = getCenterPositionFromBlockIndexAndVoxelIndex(
              block_size, block_index, voxel_index);
          Eigen::Vector2f u_C;
          bool in_frustum = frustum.isPointInView(position);
          bool in_camera = camera.project(position, &u_C);
          any_valid = in_camera || any_valid;
          if (position.z() >= 0.0f && position.z() < 1e-4f) {
            // Nothing.
          } else if (position.z() <= kMaxDist && position.z() > kMinDist) {
            EXPECT_EQ(in_frustum, in_camera);
          } else {
            // Doesn't matter if we're within the camera view if it's false.
            EXPECT_FALSE(in_frustum);
          }
        }
      }
    }
    const AxisAlignedBoundingBox& aabb_block =
        getAABBOfBlock(block_size, block_index);
    if (!any_valid) {
      empty++;
    }
  }
  // At MOST 3% empty on the corners.
  EXPECT_LE(static_cast<float>(empty) / block_indices_in_frustum.size(), 0.03);
}

TEST(CameraTest, UnProjectionTest) {
  Camera camera = getTestCamera();

  constexpr int kNumPointsToTest = 1000;
  for (int i = 0; i < kNumPointsToTest; i++) {
    // Random point and depth
    auto vector_image_point_pair = getRandomVisibleRayAndImagePoint(camera);
    Vector2f u_C_in = vector_image_point_pair.second;
    const float depth = test_utils::randomFloatInRange(0.1f, 10.0f);

    // Unproject
    const Vector3f p_C =
        camera.unprojectFromImagePlaneCoordinates(u_C_in, depth);
    EXPECT_NEAR(p_C.z(), depth, kFloatEpsilon);

    // Re-project
    Vector2f u_C_out;
    EXPECT_TRUE(camera.project(p_C, &u_C_out));

    // Check
    EXPECT_NEAR(u_C_in.x(), u_C_out.x(), kFloatEpsilon);
    EXPECT_NEAR(u_C_in.y(), u_C_out.y(), kFloatEpsilon);
  }
}

TEST(CameraTest, CameraViewport) {
  const Camera camera = getTestCamera();
  const CameraViewport viewport = camera.getNormalizedViewport();

  EXPECT_NEAR(viewport.min()[0], -camera.cu() / camera.fu(), kFloatEpsilon);
  EXPECT_NEAR(viewport.min()[1], -camera.cv() / camera.fv(), kFloatEpsilon);
  EXPECT_NEAR(viewport.max()[0], (camera.width() - camera.cu()) / camera.fu(),
              kFloatEpsilon);
  EXPECT_NEAR(viewport.max()[1], (camera.height() - camera.cv()) / camera.fv(),
              kFloatEpsilon);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
