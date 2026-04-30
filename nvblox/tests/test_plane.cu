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
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/geometry/plane.h"

#include "nvblox/tests/utils.h"

using namespace nvblox;

constexpr float kEps = 1.0e-4;

namespace {

constexpr int kPlaneAmbientDimension = 3;
typedef Eigen::Hyperplane<float, kPlaneAmbientDimension> EigenPlane;

}  // namespace

__global__ void extractComponents(const Plane plane, Vector3f* normal_ptr,
                                  float* d_ptr) {
  // Extract the components in a single thread.
  *normal_ptr = plane.normal();
  *d_ptr = plane.d();
}

__global__ void projectOnGpu(const Plane plane, Vector3f point,
                             float* signed_distance_ptr,
                             Vector3f* projection_ptr) {
  // Extract the components in a single thread.
  *signed_distance_ptr = plane.signedDistance(point);
  *projection_ptr = plane.project(point);
}

TEST(PlaneTest, ExtractComponentsOnGpu) {
  auto cuda_stream = std::make_shared<CudaStreamOwning>();

  const Vector3f normal(0.0f, 0.0f, 1.0f);
  const float d = 0.0f;
  Plane plane(normal, d);

  // Check that we get out what we put in - CPU
  EXPECT_NEAR((plane.normal() - normal).norm(), 0.0f, kEps);
  EXPECT_NEAR(plane.d(), 0.0f, kEps);

  // Check that we get out what we put in - GPU
  auto normal_gpu = make_unified<Vector3f>(MemoryType::kUnified);
  auto d_gpu = make_unified<float>(MemoryType::kUnified);
  extractComponents<<<1, 1, 0, *cuda_stream>>>(plane, normal_gpu.get(),
                                               d_gpu.get());
  cuda_stream->synchronize();
  EXPECT_NEAR((plane.normal() - *normal_gpu).norm(), 0.0f, kEps);
  EXPECT_NEAR(*d_gpu, 0.0f, kEps);
}

TEST(PlaneTest, Projection) {
  // Horizontal plane.
  Plane horizontal_plane(Vector3f(0.0f, 0.0f, 1.0f), 0.0f);

  // Project: Known point.
  Vector3f test_point = Vector3f(1.0f, 1.0f, 1.0f);
  Vector3f closest_point = horizontal_plane.project(test_point);
  EXPECT_NEAR((closest_point - Vector3f(1.0f, 1.0f, 0.0f)).norm(), 0.0f, kEps);

  // Random planes: Compare against Eigen.
  constexpr int kNumTests = 10000;
  for (int i = 0; i < kNumTests; i++) {
    // Random plane parameters
    const Vector3f random_unit_vector = test_utils::getRandomUnitVector3f();
    const Vector3f random_plane_point =
        test_utils::getRandomVector3fInRange(-100.0f, 100.0f);

    Plane plane(random_unit_vector, random_plane_point);
    EigenPlane eigen_plane(random_unit_vector, random_plane_point);

    // Random point to test
    test_point = test_utils::getRandomVector3fInRange(-100.0f, 100.0f);

    // Projection of the test point onto the plane.
    closest_point = plane.project(test_point);
    const Vector3f eigen_closest_point = eigen_plane.projection(test_point);
    EXPECT_NEAR((closest_point - eigen_closest_point).norm(), 0.0f, kEps);

    // Signed distance of the point to the plane.
    const float distance = plane.signedDistance(test_point);
    const float eigen_distance = eigen_plane.signedDistance(test_point);
    EXPECT_NEAR(distance, eigen_distance, kEps);
  }
}

TEST(PlaneTest, FromNormalAndPoint) {
  Vector3f normal = Vector3f(0.0f, 0.0f, 1.0f);
  Vector3f point = Vector3f(0.0f, 0.0f, 0.0f);
  Plane plane(normal, point);
  EXPECT_NEAR((plane.normal() - normal).norm(), 0.0f, kEps);
  EXPECT_NEAR(plane.d(), 0.0f, kEps);

  normal = Vector3f(-1.0f, 0.0f, 1.0f).normalized();
  plane = Plane(normal, point);
  EXPECT_NEAR((plane.normal() - normal).norm(), 0.0f, kEps);
  EXPECT_NEAR(plane.d(), 0.0f, kEps);

  normal = Vector3f(1.0f, 0.0f, 1.0f).normalized();
  point = Vector3f(0.0f, 0.0f, 1.0f);
  plane = Plane(normal, point);
  EigenPlane eigen_plane(normal, point);
  EXPECT_NEAR((plane.normal() - normal).norm(), 0.0f, kEps);
  EXPECT_NEAR(plane.d(), -sqrt(1.0f / 2.0f), kEps);
  EXPECT_NEAR(plane.d(), eigen_plane.offset(), kEps);
}

TEST(PlaneTest, FromThreePoints) {
  // Known points
  Vector3f test_point_a = Vector3f(0.0f, -1.0f, 2.0f);
  Vector3f test_point_b = Vector3f(2.0f, 1.0f, 1.0f);
  Vector3f test_point_c = Vector3f(12.0f, 0.0f, 1.0f);
  Plane current_plane;
  const auto plane_result = Plane::planeFromPoints(
      test_point_a, test_point_b, test_point_c, &current_plane);
  EXPECT_TRUE(plane_result);

  const Vector3f expected_normal = Vector3f(-0.0413449, -0.413449, -0.909588);
  const float expected_d = 1.405726;
  EXPECT_NEAR((current_plane.normal() - expected_normal).norm(), 0.0f, kEps);
  EXPECT_NEAR(current_plane.d(), expected_d, kEps);

  // Non- distinct points
  EXPECT_FALSE(Plane::planeFromPoints(test_point_a, test_point_a, test_point_a,
                                      &current_plane));

  // Collinear points
  test_point_b = Vector3f(0.0f, -10.0f, 2.0f);
  test_point_c = Vector3f(0.0f, -100.0f, 2.0f);
  EXPECT_FALSE(Plane::planeFromPoints(test_point_a, test_point_b, test_point_c,
                                      &current_plane));
}

TEST(PlaneTest, ProjectionOnGpu) {
  auto cuda_stream = std::make_shared<CudaStreamOwning>();

  constexpr int kNumTests = 1000;
  for (int i = 0; i < kNumTests; i++) {
    Plane plane(test_utils::getRandomUnitVector3f(),
                test_utils::randomFloatInRange(-100.0, 100.0));
    Vector3f point = test_utils::getRandomVector3fInRange(-100.0f, 100.0f);

    // Project on GPU
    auto distance_gpu = make_unified<float>(MemoryType::kUnified);
    auto projection_gpu = make_unified<Vector3f>(MemoryType::kUnified);
    projectOnGpu<<<1, 1, 0, *cuda_stream>>>(plane, point, distance_gpu.get(),
                                            projection_gpu.get());
    cuda_stream->synchronize();

    // Project on CPU
    const float distance_cpu = plane.signedDistance(point);
    const Vector3f projection_cpu = plane.project(point);

    // Check
    EXPECT_NEAR(distance_cpu, *distance_gpu, kEps);
    EXPECT_NEAR((*projection_gpu - projection_cpu).norm(), 0, kEps);
  }
}

TEST(PlaneTest, HeightFromXY) {
  // Z-up, through 0,0,0.
  Plane plane({0.0f, 0.0f, 1.0f}, Vector3f(0.0f, 0.0f, 0.0f));
  float height = plane.getHeightAtXY({0.0f, 0.0f});
  EXPECT_NEAR(height, 0.0f, kEps);
  height = plane.getHeightAtXY({1.0f, 1.0f});
  EXPECT_NEAR(height, 0.0f, kEps);

  // Z-up, through 0,0,1.
  plane = Plane({0.0f, 0.0f, 1.0f}, Vector3f(0.0f, 0.0f, 1.0f));
  height = plane.getHeightAtXY({0.0f, 0.0f});
  EXPECT_NEAR(height, 1.0f, kEps);
  height = plane.getHeightAtXY({1.0f, 1.0f});
  EXPECT_NEAR(height, 1.0f, kEps);

  // 45 degrees through 0,0,0
  plane = Plane(Vector3f(-1.0f, 0.0f, 1.0f).normalized(),
                Vector3f(0.0f, 0.0f, 0.0f));
  height = plane.getHeightAtXY({0.0f, 0.0f});
  EXPECT_NEAR(height, 0.0f, kEps);
  height = plane.getHeightAtXY({1.0f, 1.0f});
  EXPECT_NEAR(height, 1.0f, kEps);

  // 45 degrees through 0,0,1
  plane = Plane(Vector3f(-1.0f, 0.0f, 1.0f).normalized(),
                Vector3f(0.0f, 0.0f, 1.0f));
  height = plane.getHeightAtXY({0.0f, 0.0f});
  EXPECT_NEAR(height, 1.0f, kEps);
  height = plane.getHeightAtXY({1.0f, 1.0f});
  EXPECT_NEAR(height, 2.0f, kEps);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
