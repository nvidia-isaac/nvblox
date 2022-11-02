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

#include "nvblox/core/image.h"
#include "nvblox/core/oslidar.h"
#include "nvblox/core/types.h"

#include <iostream>
#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-4;
class oslidarTest : public ::testing::Test {
 protected:
  oslidarTest() {}
};

class ParameterizedoslidarTest
    : public oslidarTest,
      public ::testing::WithParamInterface<std::tuple<int, int, float, float>> {
 protected:
  // Yo dawg I heard you like params
};

TEST_P(ParameterizedoslidarTest, Extremes) {
  // oslidar params
  const auto params = GetParam();
  const int num_azimuth_divisions = std::get<0>(params);
  const int num_elevation_divisions = std::get<1>(params);
  const float vertical_fov_deg = std::get<2>(params);
  const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;
  const float horizontal_fov_deg = std::get<3>(params);
  const float horizontal_fov_rad = horizontal_fov_deg * M_PI / 180.0f;

  DepthImage depth_image(num_elevation_divisions, num_azimuth_divisions);
  DepthImage z_image(num_elevation_divisions, num_azimuth_divisions);
  OSLidar oslidar(num_azimuth_divisions, num_elevation_divisions,
                  horizontal_fov_rad, vertical_fov_rad, depth_image, z_image);

  //-------------------
  // Elevation extremes
  //-------------------
  float azimuth_center_pixel =
      static_cast<float>(num_azimuth_divisions - 1) / 2.0f;
  float elevation_center_pixel =
      static_cast<float>(num_elevation_divisions - 1) / 2.0f;

  float elevation_top_pixel = 0.0f;
  float elevation_bottom_pixel =
      static_cast<float>(num_elevation_divisions) - 1;

  const float x_dist = 10;
  const float z_dist = x_dist * tan(vertical_fov_rad / 2.0f);

  // Top beam
  Vector2f u_C;
  Vector3f p = Vector3f(x_dist, 0.0, z_dist);
  EXPECT_TRUE(oslidar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_center_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_top_pixel, kFloatEpsilon);

  // Center
  p = Vector3f(x_dist, 0.0f, 0.0f);
  EXPECT_TRUE(oslidar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_center_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_center_pixel, kFloatEpsilon);

  // Bottom beam
  p = Vector3f(x_dist, 0.0f, -z_dist);
  EXPECT_TRUE(oslidar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_center_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_bottom_pixel, kFloatEpsilon);

  //-----------------
  // Azimuth extremes
  //-----------------

  float azimuth_left_pixel = 0.0;
  float azimuth_quarter_pixel =
      (azimuth_center_pixel - azimuth_left_pixel) / 2.0f + azimuth_left_pixel;
  float azimuth_three_quarter_pixel =
      (azimuth_center_pixel - azimuth_left_pixel) / 2.0f + azimuth_center_pixel;

  // Backwards
  p = Vector3f(-1.0, 0.0, 0.0);
  EXPECT_TRUE(oslidar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_left_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_center_pixel, kFloatEpsilon);

  // Right
  p = Vector3f(0.0, -1.0, 0.0);
  EXPECT_TRUE(oslidar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_three_quarter_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_center_pixel, kFloatEpsilon);

  // Forwards
  p = Vector3f(1.0, 0.0, 0.0);
  EXPECT_TRUE(oslidar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_center_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_center_pixel, kFloatEpsilon);

  // Left
  p = Vector3f(0.0, 1.0, 0.0);
  EXPECT_TRUE(oslidar.project(p, &u_C));
  EXPECT_NEAR(u_C.x(), azimuth_quarter_pixel, kFloatEpsilon);
  EXPECT_NEAR(u_C.y(), elevation_center_pixel, kFloatEpsilon);
}

TEST_P(ParameterizedoslidarTest, SphereTest) {
  // oslidar params
  const auto params = GetParam();
  const int num_azimuth_divisions = std::get<0>(params);
  const int num_elevation_divisions = std::get<1>(params);
  const float vertical_fov_deg = std::get<2>(params);
  const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;
  const float horizontal_fov_deg = std::get<3>(params);
  const float horizontal_fov_rad = horizontal_fov_deg * M_PI / 180.0f;

  DepthImage depth_image(num_elevation_divisions, num_azimuth_divisions);
  DepthImage z_image(num_elevation_divisions, num_azimuth_divisions);
  OSLidar oslidar(num_azimuth_divisions, num_elevation_divisions,
                  horizontal_fov_rad, vertical_fov_rad, depth_image, z_image);
  EXPECT_EQ(z_image.rows(), num_elevation_divisions);
  EXPECT_EQ(z_image.cols(), num_azimuth_divisions);

  // Pointcloud
  Eigen::MatrixX3f pointcloud(num_azimuth_divisions * num_elevation_divisions,
                              3);
  Eigen::MatrixXf desired_image(num_elevation_divisions, num_azimuth_divisions);

  // Construct a pointcloud of a points at random distances.
  const float azimuth_increments_rad = 2 * M_PI / (num_azimuth_divisions - 1);
  const float polar_increments_rad =
      vertical_fov_rad / (num_elevation_divisions - 1);
  const float half_vertical_fov_rad = vertical_fov_rad / 2.0f;

  int point_idx = 0;
  for (int az_idx = 0; az_idx < num_azimuth_divisions; az_idx++) {
    for (int el_idx = 0; el_idx < num_elevation_divisions; el_idx++) {
      // const float azimuth_rad = az_idx * azimuth_increments_rad - M_PI;
      // const float polar_rad =
      //     el_idx * polar_increments_rad + (M_PI / 2.0 -
      //     half_vertical_fov_rad);
      const float azimuth_rad = M_PI - az_idx * azimuth_increments_rad;
      const float polar_rad =
          el_idx * polar_increments_rad - half_vertical_fov_rad;
      // 0.0 <= el_idx * polar_increments_rad <= vertical_fov_rad
      // -half_vertical_fov_rad <= polar_rad <= half_vertical_fov_rad

      constexpr float max_depth = 10.0;
      constexpr float min_depth = 1.0;
      const float distance =
          test_utils::randomFloatInRange(min_depth, max_depth);

      const float x = distance * cos(polar_rad) * cos(azimuth_rad);
      const float y = distance * cos(polar_rad) * sin(azimuth_rad);
      const float z = distance * sin(polar_rad);

      pointcloud(point_idx, 0) = x;
      pointcloud(point_idx, 1) = y;
      pointcloud(point_idx, 2) = z;

      // from bottom to top
      desired_image(num_elevation_divisions - el_idx - 1, az_idx) = distance;

      point_idx++;
    }
  }

  // Project the pointcloud to a depth image
  Eigen::MatrixXf reprojected_image(num_elevation_divisions,
                                    num_azimuth_divisions);
  for (int point_idx = 0; point_idx < pointcloud.rows(); point_idx++) {
    // Projection
    Vector2f u_C_float;
    EXPECT_TRUE(oslidar.project(pointcloud.row(point_idx), &u_C_float));
    Index2D u_C_int;
    EXPECT_TRUE(oslidar.project(pointcloud.row(point_idx), &u_C_int));

    // Check that this is at the center of a pixel
    Vector2f corner_dist = u_C_float - u_C_float.array().round().matrix();
    constexpr float kReprojectionEpsilon = 0.001;
    EXPECT_NEAR(corner_dist.x(), 0.0f, kReprojectionEpsilon);
    EXPECT_NEAR(corner_dist.y(), 0.0f, kReprojectionEpsilon);

    // Add to depth image
    reprojected_image(u_C_int.y(), u_C_int.x()) =
        pointcloud.row(point_idx).norm();
  }

  const Eigen::MatrixXf error_image =
      desired_image.array() - reprojected_image.array();
  float max_error = error_image.rowwise().mean().mean();
  EXPECT_NEAR(max_error, 0.0, kFloatEpsilon);
}

TEST_P(ParameterizedoslidarTest, OutOfBoundsTest) {
  // oslidar params
  const auto params = GetParam();
  const int num_azimuth_divisions = std::get<0>(params);
  const int num_elevation_divisions = std::get<1>(params);
  const float vertical_fov_deg = std::get<2>(params);
  const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;
  const float horizontal_fov_deg = std::get<3>(params);
  const float horizontal_fov_rad = horizontal_fov_deg * M_PI / 180.0f;

  DepthImage depth_image(num_elevation_divisions, num_azimuth_divisions);
  DepthImage z_image(num_elevation_divisions, num_azimuth_divisions);
  OSLidar oslidar(num_azimuth_divisions, num_elevation_divisions,
                  horizontal_fov_rad, vertical_fov_rad, depth_image, z_image);

  // Outside on top and bottom
  const float rads_per_pixel_elevation =
      vertical_fov_rad / static_cast<float>(num_elevation_divisions - 1);
  const float x_dist = 10;
  const float z_dist = x_dist * tan(vertical_fov_rad / 2.0f +
                                    rads_per_pixel_elevation / 2.0f + 0.1);

  Vector2f u_C_float;
  EXPECT_FALSE(oslidar.project(Vector3f(x_dist, 0.0f, z_dist), &u_C_float));
  EXPECT_FALSE(oslidar.project(Vector3f(x_dist, 0.0f, -z_dist), &u_C_float));
  Index2D u_C_int;
  EXPECT_FALSE(oslidar.project(Vector3f(x_dist, 0.0f, z_dist), &u_C_int));
  EXPECT_FALSE(oslidar.project(Vector3f(x_dist, 0.0f, -z_dist), &u_C_int));

  // Make a bunch of points with only azimuth and check they all pass
  for (int i = 0; i < 1000; i++) {
    const float theta = test_utils::randomFloatInRange(-M_PI - 0.1, M_PI + 0.1);
    const float radius = 10.0f;
    const float x = radius * cos(theta);
    const float y = radius * cos(theta);
    const float z = 0;

    EXPECT_TRUE(oslidar.project(Vector3f(x, y, z), &u_C_float));
    EXPECT_TRUE(oslidar.project(Vector3f(x, y, z), &u_C_int));
  }
}

TEST_P(ParameterizedoslidarTest, PixelToRayExtremes) {
  //   // oslidar params
  //   const auto params = GetParam();
  //   const int num_azimuth_divisions = std::get<0>(params);
  //   const int num_elevation_divisions = std::get<1>(params);
  //   const float vertical_fov_deg = std::get<2>(params);
  //   const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;
  //   const float half_vertical_fov_rad = vertical_fov_rad / 2.0;

  //   DepthImagez_image(epthImage(num_elevation_divisions,
  //   num_azimuth_divisions); OSLidar oslidar(num_azimuth_divisions,
  //   num_elevation_divisions,
  //                    z_image);vertical_fov, **
  //   // Special pixels to use
  //   const float middle_elevation_pixel = (num_elevation_divisions - 1) /
  // 2;
  //   const float middle_azimuth_pixel = num_azimuth_divisions / 2;
  //   const float right_azimuth_pixel = num_azimuth_divisions / 4;
  //   const float left_azimuth_pixel = 3 * num_azimuth_divisions / 4;

  //   const float middle_elevation_coords = middle_elevation_pixel + 0.5f;
  //   const float middle_azimuth_coords = middle_azimuth_pixel + 0.5f;
  //   const float right_azimuth_coords = right_azimuth_pixel + 0.5f;
  //   const float left_azimuth_coords = left_azimuth_pixel + 0.5f;

  //   //-----------------
  //   // Azimuth extremes
  //   //-----------------

  //   // NOTE(alexmillane): We get larger than I would expect errors on some
  // of
  //   the
  //   // components. I had to turn up the allowable error a bit.
  //   constexpr float kAllowableVectorError = 0.02;

  //   // Backwards
  //   Vector3f v_C = oslidar.vectorFromImagePlaneCoordinates(
  //       Vector2f(0.5, middle_elevation_coords));
  //   EXPECT_NEAR(v_C.x(), -1.0f, kAllowableVectorError);
  //   EXPECT_NEAR(v_C.y(), 0.0f, kAllowableVectorError);
  //   EXPECT_NEAR(v_C.z(), 0.0f, kAllowableVectorError);

  //   // Forwards
  //   v_C = oslidar.vectorFromImagePlaneCoordinates(
  //       Vector2f(middle_azimuth_coords, middle_elevation_coords));
  //   EXPECT_NEAR(v_C.x(), 1.0f, kAllowableVectorError);
  //   EXPECT_NEAR(v_C.y(), 0.0f, kAllowableVectorError);
  //   EXPECT_NEAR(v_C.z(), 0.0f, kAllowableVectorError);

  //   // Right
  //   v_C = oslidar.vectorFromImagePlaneCoordinates(
  //       Vector2f(right_azimuth_coords, middle_elevation_coords));
  //   EXPECT_NEAR(v_C.x(), 0.0f, kAllowableVectorError);
  //   EXPECT_NEAR(v_C.y(), -1.0f, kAllowableVectorError);
  //   EXPECT_NEAR(v_C.z(), 0.0f, kAllowableVectorError);

  //   // Left
  //   v_C = oslidar.vectorFromImagePlaneCoordinates(
  //       Vector2f(left_azimuth_coords, middle_elevation_coords));
  //   EXPECT_NEAR(v_C.x(), 0.0f, kAllowableVectorError);
  //   EXPECT_NEAR(v_C.y(), 1.0f, kAllowableVectorError);
  //   EXPECT_NEAR(v_C.z(), 0.0f, kAllowableVectorError);

  //   //-----------------
  //   // Elevation extremes
  //   //-----------------
  //   const Vector3f top_of_elevation_forward_vector(
  //       cos(half_vertical_fov_rad), 0.0f, sin(half_vertical_fov_rad));
  //   const Vector3f bottom_of_elevation_forward_vector(
  //       cos(half_vertical_fov_rad), 0.0f, -sin(half_vertical_fov_rad));

  //   v_C = oslidar.vectorFromImagePlaneCoordinates(
  //       Vector2f(middle_azimuth_coords, 0.5));
  //   EXPECT_NEAR(v_C.dot(top_of_elevation_forward_vector), 1.0f,
  //   kFloatEpsilon);

  //   v_C = oslidar.vectorFromImagePlaneCoordinates(
  //       Vector2f(middle_azimuth_coords, num_elevation_divisions - 1 +
  // 0.5));
  //   EXPECT_NEAR(v_C.dot(bottom_of_elevation_forward_vector), 1.0f,
  //   kFloatEpsilon);
}

// clang-format off
INSTANTIATE_TEST_CASE_P(
    ParameterizedoslidarTests, ParameterizedoslidarTest, ::testing::Values(
      std::tuple<int, int, float, float>(2048, 128, 45.0f, 360.0f)));
// clang-format on

TEST_P(ParameterizedoslidarTest, RandomPixelRoundTrips) {
  // Make sure this is deterministic.
  std::srand(0);

  const auto params = GetParam();
  const int num_azimuth_divisions = std::get<0>(params);
  const int num_elevation_divisions = std::get<1>(params);
  const float vertical_fov_deg = std::get<2>(params);
  const float vertical_fov_rad = vertical_fov_deg * M_PI / 180.0f;
  const float horizontal_fov_deg = std::get<3>(params);
  const float horizontal_fov_rad = horizontal_fov_deg * M_PI / 180.0f;

  DepthImage depth_image(num_elevation_divisions, num_azimuth_divisions);
  DepthImage z_image(num_elevation_divisions, num_azimuth_divisions);
  OSLidar oslidar(num_azimuth_divisions, num_elevation_divisions,
                  horizontal_fov_rad, vertical_fov_rad, depth_image, z_image);

  // TODO(jjiao): The projection and unprojection of points of OS lidar is
  // uninvertible
  // Test a large number of points const int kNumberOfPointsToTest
  // = 10000; for (int i = 0; i < kNumberOfPointsToTest; i++) {
  //   // Random point on the image plane
  //   const Vector2f u_C(test_utils::randomFloatInRange(
  //                          0.0f,
  // static_cast<float>(num_azimuth_divisions)),
  //                      test_utils::randomFloatInRange(
  //                          0.0f,
  //                          static_cast<float>(num_elevation_divisions)));
  //   // Pixel -> Ray
  //   const Vector3f v_C = oslidar.vectorFromImagePlaneCoordinates(u_C);
  //   // Randomly scale ray
  //   const Vector3f p_C = v_C *
  // test_utils::randomFloatInRange(0.1f, 10.0f);

  //   // Project back to image plane
  //   Vector2f u_C_reprojected;
  //   EXPECT_TRUE(oslidar.project(p_C, &u_C_reprojected));
  //   // Check we get back to where we started
  //   constexpr float kAllowableReprojectionError = 0.001;
  //   EXPECT_NEAR((u_C_reprojected - u_C).maxCoeff(), 0.0f,
  //               kAllowableReprojectionError);
  // }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
