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
#include <gtest/gtest.h>
#include <vector>

#include "nvblox/core/types.h"
#include "nvblox/geometry/plane.h"
#include "nvblox/geometry/transforms.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kTransformEps = 1.0e-3f;

TEST(TransformTest, ComputeTransformToAlignPlaneToZ0) {
  // Test 1: Horizontal plane at z=1
  {
    Plane horizontal_plane(Vector3f(0.0f, 0.0f, 1.0f), -1.0f);
    Transform T = computeTransformToAlignPlaneToZ0(horizontal_plane);

    // Translation should move z=1 to z=0
    EXPECT_NEAR(T.translation().z(), -1.0f, kTransformEps);

    // Verify a point on the plane (at z=1) ends up at z=0
    Vector3f point_on_plane(1.0f, 2.0f, 1.0f);
    Vector3f transformed_point = T * point_on_plane;
    EXPECT_NEAR(transformed_point.z(), 0.0f, kTransformEps);

    // Verify a point above the plane remains above after transformation
    Vector3f point_above(1.0f, 2.0f, 2.0f);
    EXPECT_GT(horizontal_plane.signedDistance(point_above), 0.0f);
    transformed_point = T * point_above;
    EXPECT_GT(transformed_point.z(), 0.0f);

    // Verify a point below the plane remains below after transformation
    Vector3f point_below(1.0f, 2.0f, 0.0f);
    EXPECT_LT(horizontal_plane.signedDistance(point_below), 0.0f);
    transformed_point = T * point_below;
    EXPECT_LT(transformed_point.z(), 0.0f);
  }

  // Test 2: Tilted plane with offset
  {
    Vector3f normal = Vector3f(1.0f, 0.0f, 1.0f).normalized();
    Plane tilted_plane(normal, Vector3f(0.0f, 0.0f, 2.0f));
    Transform T = computeTransformToAlignPlaneToZ0(tilted_plane);

    // Verify that points on the plane end up at z=0
    constexpr int kNumTestPoints = 10;
    for (int i = 0; i < kNumTestPoints; i++) {
      // Generate a random point on the plane
      Vector3f random_point =
          test_utils::getRandomVector3fInRange(-10.0f, 10.0f);
      Vector3f point_on_plane = tilted_plane.project(random_point);

      // Transform the point
      Vector3f transformed_point = T * point_on_plane;

      // The transformed point should be at z=0 (within tolerance)
      EXPECT_NEAR(std::abs(transformed_point.z()), 0.0f, kTransformEps);
    }
  }

  // Test 3: Plane normal pointing down (opposite z-axis)
  {
    Plane downward_plane(Vector3f(0.0f, 0.0f, -1.0f), -1.0f);
    Transform T = computeTransformToAlignPlaneToZ0(downward_plane);

    // Verify a point on the plane (at z=-1) ends up at z=0
    Vector3f point_on_plane(1.0f, 2.0f, -1.0f);
    EXPECT_NEAR(downward_plane.signedDistance(point_on_plane), 0.0f,
                kTransformEps);
    Vector3f transformed_point = T * point_on_plane;
    EXPECT_NEAR(transformed_point.z(), 0.0f, kTransformEps);

    // Verify a point above ground, remains above after transformation
    Vector3f point_above(1.0f, 2.0f, 2.0f);
    transformed_point = T * point_above;
    EXPECT_GT(transformed_point.z(), 0.0f);

    // Verify a point below ground, remains below after transformation
    Vector3f point_below(1.0f, 2.0f, -2.0f);
    transformed_point = T * point_below;
    EXPECT_LT(transformed_point.z(), 0.0f);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
