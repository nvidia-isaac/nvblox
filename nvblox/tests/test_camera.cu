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

#include "nvblox/core/internal/error_check.h"
#include "nvblox/core/types.h"
#include "nvblox/geometry/bounding_boxes.h"
#include "nvblox/io/image_io.h"
#include "nvblox/primitives/primitives.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/image.h"
#include "nvblox/tests/sensor_fixture.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;
// TODO: Decide where to put test epsilons
// NOTE(alexmillane): I had to crank this up slightly to get things to
// pass... I guess this is just floating point errors accumulating?
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

float randomScale() {
  constexpr float kS = 0.1;
  return test_utils::randomFloatInRange(1.0 - kS, 1.0 + kS);
}
Camera getCameraRandomDistortion() {
  // In order to get a realistic and non-degenerate camera, we start with a
  // known calibration which is perturbed.
  const Camera camera = test_utils::getOrbecCamera();
  const auto dist = camera.distortion_params().value();
  const float fu = camera.fu() * randomScale();
  const float fv = camera.fv() * randomScale();
  const float cu = camera.cu() * randomScale();
  const float cv = camera.cv() * randomScale();
  const float k1 = dist.radial.k1 * randomScale();
  const float k2 = dist.radial.k2 * randomScale();
  const float k3 = dist.radial.k3 * randomScale();
  const float k4 = dist.radial.k4 * randomScale();
  const float k5 = dist.radial.k5 * randomScale();
  const float k6 = dist.radial.k6 * randomScale();
  const float p1 =
      camera.distortion_params().value().tangential.p1 * randomScale();
  const float p2 =
      camera.distortion_params().value().tangential.p2 * randomScale();

  return Camera(
      fu, fv, cu, cv, camera.width(), camera.height(),
      RadialTangentialDistortionParams{{k1, k2, k3, k4, k5, k6}, {p1, p2}});
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

TEST(CameraTest, InvalidPointProjections) {
  const Camera camera = getTestCamera();
  Vector2f u_C;

  // Test that valid point projects successfully
  // Point at center of camera view
  Vector3f valid_point(0.0f, 0.0f, 5.0f);
  EXPECT_TRUE(camera.project(valid_point, &u_C))
      << "Valid point at camera center should project";

  // Test invalid values (NaN, +inf, -inf) in each coordinate
  const std::vector<float> invalid_values = {
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(),
      -std::numeric_limits<float>::infinity()};

  for (const float invalid_val : invalid_values) {
    for (int coord = 0; coord < 3; ++coord) {
      Vector3f invalid_point(0.0f, 0.0f, 5.0f);
      invalid_point[coord] = invalid_val;
      EXPECT_FALSE(camera.project(invalid_point, &u_C))
          << "Point with invalid value " << invalid_val << " at coord " << coord
          << " should fail to project";
    }
  }

  // Test one point behind camera (negative z)
  Vector3f point_behind_camera(1.0f, 1.0f, -1.0f);
  EXPECT_FALSE(camera.project(point_behind_camera, &u_C))
      << "Point behind camera " << point_behind_camera.transpose()
      << " should fail to project";

  // Test points with z less than default min_depth
  constexpr float kDefaultMinDepth = Camera::kDefaultMinProjectionDepth;
  Vector3f too_close(1.0f, 1.0f, kDefaultMinDepth / 2.0f);
  EXPECT_FALSE(camera.project(too_close, &u_C))
      << "Point closer than min_depth should fail to project";

  // Test that point exactly at min_depth projects (boundary case)
  Vector3f at_min_depth(0.0f, 0.0f, kDefaultMinDepth);
  EXPECT_TRUE(camera.project(at_min_depth, &u_C))
      << "Point at exactly min_depth should project";
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

// Check that radial distortion is correct for random pixels.
TEST(CameraTest, RadialDistortionScale_PixelsAreDistorted) {
  constexpr int kNumIterations = 100000;
  for (int i = 0; i < kNumIterations; i++) {
    const Camera camera = getCameraRandomDistortion();

    const auto dist = camera.distortion_params().value();
    const float k1 = dist.radial.k1;
    const float k2 = dist.radial.k2;
    const float k3 = dist.radial.k3;
    const float k4 = dist.radial.k4;
    const float k5 = dist.radial.k5;
    const float k6 = dist.radial.k6;

    const float r2 = test_utils::randomFloatInRange(0.F, 10.F);
    const float actual = radialDistortionScale<float>(
        r2, camera.distortion_params().value().radial);
    const float expected = (1.F + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) /
                           (1.F + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2);
    EXPECT_NEAR(actual, expected, kFloatEpsilon);
  }
}

// Center pixel has no tangential distortion.
TEST(CameraTest, TangentialDistortion_CenterPixelNotDistorted) {
  const Vector2f u_normalized(0.F, 0.F);
  const float r2 = 0.F;
  const Vector2f tangential_distortion = applyTangentialDistortion<float>(
      u_normalized, r2, TangentialDistortionParams{0.1, 0.2});

  EXPECT_TRUE(tangential_distortion.isApprox(Vector2f::Zero(), kFloatEpsilon));
}

// Check that tangential distortion is correct for random pixels.
TEST(CameraTest, TangentialDistortion_PixelsAreDistorted) {
  constexpr int kNumIterations = 100000;
  for (int i = 0; i < kNumIterations; i++) {
    const Camera camera = getCameraRandomDistortion();
    const float p1 = camera.distortion_params().value().tangential.p1;
    const float p2 = camera.distortion_params().value().tangential.p2;
    const float x = test_utils::randomFloatInRange(-2.F, 2.F);
    const float y = test_utils::randomFloatInRange(-2.F, 2.F);
    const Vector2f u_normalized(x, y);
    const float r2 = x * x + y * y;

    const Vector2f actual = applyTangentialDistortion<float>(
        u_normalized, r2, camera.distortion_params().value().tangential);

    const float expected_x = 2.F * p1 * x * y + p2 * (r2 + 2.F * x * x);
    const float expected_y = 2.F * p2 * x * y + p1 * (r2 + 2.F * y * y);
    const Vector2f expected(expected_x, expected_y);

    EXPECT_TRUE(actual.isApprox(expected, kFloatEpsilon));
  }
}

TEST(CameraTest, DistortionUndistortion_P3dRoundtrip) {
  constexpr int kNumIterations = 100000;
  for (int i = 0; i < kNumIterations; i++) {
    const Camera camera_with_distortion = getCameraRandomDistortion();

    Vector3f ray_C;
    Vector2f u_C;
    std::tie(ray_C, u_C) =
        getRandomVisibleRayAndImagePoint(camera_with_distortion);

    // Create a 3D point at random depth
    const Vector3f p_gt = test_utils::randomFloatInRange(1.0, 1000.0) * ray_C;

    // Project with distortion
    Vector2f u_distorted_C;
    EXPECT_TRUE(camera_with_distortion.project(p_gt, &u_distorted_C));

    // Unproject back (should handle undistortion)
    Vector3f p_undistorted_C =
        camera_with_distortion.unprojectFromImagePlaneCoordinates(u_distorted_C,
                                                                  p_gt.z());

    // The undistorted point should match the original (within tolerance)
    EXPECT_TRUE(p_gt.isApprox(p_undistorted_C, kFloatEpsilon * p_gt.z()));
  }
}

TEST(CameraTest, DistortionUndistortion_P2dRoundtrip) {
  constexpr int kNumIterations = 100000;
  for (int i = 0; i < kNumIterations; i++) {
    const Camera camera_with_distortion = getCameraRandomDistortion();

    Vector2f u_px = Vector2f(
        test_utils::randomFloatInRange(0.0, camera_with_distortion.width()),
        test_utils::randomFloatInRange(0.0, camera_with_distortion.height()));

    constexpr float kDepth = 10.0;

    const Vector3f p_C =
        camera_with_distortion.unprojectFromImagePlaneCoordinates(u_px, kDepth);

    Vector2f u_reprojected_px;
    EXPECT_TRUE(camera_with_distortion.project(p_C, &u_reprojected_px));

    EXPECT_TRUE(u_px.isApprox(u_reprojected_px, kFloatEpsilon));
  }
}

__global__ void __launch_bounds__(1024)
    cudaDistortionUndistortionP2dRoundtripKernel(const Camera camera) {
  const int x = threadIdx.x;
  const int y = blockIdx.x;

  // Skip the border pixels. They might end up slightly outside the viewport
  // when reprojecting them.
  if (x > 0 && x < camera.width() - 1 && y > 0 && y < camera.height() - 1) {
    {
      Vector2f u_px(x, y);
      const Vector3f p_C =
          camera.unprojectFromImagePlaneCoordinates(u_px, 10.0);
      Vector2f u_reprojected_px;

      NVBLOX_CHECK(camera.project(p_C, &u_reprojected_px), "Failed to project");
      NVBLOX_CHECK(u_px.isApprox(u_reprojected_px, 1e-4f),
                   "Failed to unproject");
    }
  }
}

// Roundtrip test on CUDA. iterates over all image pixels.
TEST(CameraTest, DistortionUndistortion_P2dRoundtrip_CUDA) {
  const Camera camera = getCameraRandomDistortion();

  const int num_threads = camera.width();
  const int num_blocks = camera.height();
  cudaDistortionUndistortionP2dRoundtripKernel<<<num_blocks, num_threads, 0,
                                                 CudaStreamOwning().get()>>>(
      camera);
  checkCudaErrors(cudaPeekAtLastError());
}

void renderAndWriteDepthImage(const Camera& camera,
                              const primitives::Scene& scene,
                              const std::string& filename) {
  DepthImage depth_image(camera.height(), camera.width(), MemoryType::kUnified);
  scene.generateDepthImageFromScene(camera, Transform::Identity(), 10.0,
                                    &depth_image);
  io::writeToPng(filename, depth_image);
}

TEST(CameraTest, testDerivative_dR_dr2) {
  constexpr int kNumIterations = 100000;
  for (int i = 0; i < kNumIterations; i++) {
    // Get a random camera with distortion
    const Camera camera = getCameraRandomDistortion();

    // Get a random normalized point inside the viewport
    const CameraViewport viewport = camera.getNormalizedViewport();
    const Vector2f u_normalized = Vector2f(
        test_utils::randomFloatInRange(viewport.min().x(), viewport.max().x()),
        test_utils::randomFloatInRange(viewport.min().y(), viewport.max().y()));

    // Compute the squared radius
    const double r2 = u_normalized.x() * u_normalized.x() +
                      u_normalized.y() * u_normalized.y();

    // Compute analytic derivative.
    const auto dist = camera.distortion_params().value();
    const double dR_dr2 =
        compute_dR_dr2(r2, dist.radial.k1, dist.radial.k2, dist.radial.k3,
                       dist.radial.k4, dist.radial.k5, dist.radial.k6);

    // Compute numerical derivative using finite differences.
    constexpr double kDelta = 1e-6;
    double forward = radialDistortionScale<double>(r2 + kDelta, dist.radial);
    double backward = radialDistortionScale<double>(r2 - kDelta, dist.radial);
    double dR_dr2_numerical = (forward - backward) / (2.0 * kDelta);

    // Check that the analytic and numerical derivatives are close.
    EXPECT_NEAR(dR_dr2, dR_dr2_numerical, 1E-6);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
