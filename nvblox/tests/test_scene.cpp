
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
#include <string>

#include "nvblox/io/image_io.h"
#include "nvblox/primitives/scene.h"

#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/type_indexed_store.h"
#include "nvblox/tests/sensor_fixture.h"

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-4;

template <typename SensorType>
class SceneTest : public test_utils::SensorFixture<SensorType> {
 protected:
  void SetUp() override {
    // Make the scene 6x6x3 meters big.
    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-3.0f, -3.0f, 0.0f),
                                           Vector3f(3.0f, 3.0f, 3.0f));
  }

  // sensor dependent max dist
  float getMaxDist();

  // sensor dependent check function
  void assertDepthValid(const DepthImage& depth_frame,
                        const float target_depth);

  constexpr static float kInvalidDepth = -1.F;

  // Generate a depth image of a plane
  DepthImage createPlaneDepthImge(const primitives::Plane& plane,
                                  const Transform& T_S_C,
                                  const SensorType& sensor) {
    scene_.addPrimitive(std::make_unique<primitives::Plane>(plane));

    DepthImage depth_frame(sensor.height(), sensor.width(),
                           MemoryType::kUnified);
    scene_.generateDepthImageFromScene(sensor, T_S_C, getMaxDist(),
                                       &depth_frame, kInvalidDepth);
    return depth_frame;
  }

  // A simulation scene.
  primitives::Scene scene_;
};

// Depth images store z-depth, so the distance is bounded
template <>
float SceneTest<Camera>::getMaxDist() {
  return 4.F;
}
// Lidar images store range, so distance is unbounded
template <>
float SceneTest<Lidar>::getMaxDist() {
  return 1E6;
}

// Camera depth image is expected have a constant depth for these tests.
template <>
void SceneTest<Camera>::assertDepthValid(const DepthImage& depth_frame,
                                         const float target_depth) {
  for (int lin_idx = 0; lin_idx < depth_frame.numel(); lin_idx++) {
    EXPECT_NEAR(depth_frame(lin_idx), target_depth, kFloatEpsilon);
  }
}

// Lidar depth image store ranges, so we test it differently.
template <>
void SceneTest<Lidar>::assertDepthValid(const DepthImage& depth_frame,
                                        const float target_depth) {
  int num_valid = 0;
  for (int x = 0; x < depth_frame.cols(); ++x)
    for (int y = 0; y < depth_frame.rows(); ++y) {
      Vector3f ray = this->sensor().vectorFromPixelIndices({x, y});
      // If the ray points to the upper hemishphere, we expect it to intersect
      // with the plane
      if (ray.z() > 0) {
        EXPECT_GE(depth_frame(y, x), target_depth);
        ++num_valid;
      } else {
        EXPECT_EQ(depth_frame(y, x), this->kInvalidDepth);
      }
    }
  // Since the sensor is symmetric, we expect half the points to intersect.
  EXPECT_EQ(num_valid, depth_frame.numel() / 2);
}

using SensorTypes = ::testing::Types<Camera, Lidar>;
TYPED_TEST_SUITE(SceneTest, SensorTypes);

TYPED_TEST(SceneTest, BlankMap) {
  constexpr float max_dist = 1.0;

  // Ensure that we don't get any distances back.
  Vector3f point = Vector3f::Zero();

  float dist = this->scene_.getSignedDistanceToPoint(point, max_dist);

  EXPECT_NEAR(dist, max_dist, kFloatEpsilon);

  // Ensure that we get a blank image.
  DepthImage depth_frame(this->sensor().height(), this->sensor().width(),
                         MemoryType::kUnified);
  Transform T_S_C = Transform::Identity();
  this->scene_.generateDepthImageFromScene(this->sensor(), T_S_C, max_dist,
                                           &depth_frame);

  // Check all the pixels.
  for (int lin_idx = 0; lin_idx < depth_frame.numel(); lin_idx++) {
    EXPECT_NEAR(depth_frame(lin_idx), 0.0f, kFloatEpsilon);
  }
}

TYPED_TEST(SceneTest, PlaneScene) {
  // Create a scene that's just a plane.
  // Plane at 1.0 in the positive z direction, pointing in -z.
  constexpr float kPlaneDist = 1.F;
  DepthImage depth_frame = this->createPlaneDepthImge(
      primitives::Plane(Vector3f(0.0f, 0.0, kPlaneDist), Vector3f(0, 0, -1)),
      Transform::Identity(), this->sensor());

  // Check all the pixels. Note that we do this differently depending on the
  // sensor type.
  this->assertDepthValid(depth_frame, kPlaneDist);
}

TYPED_TEST(SceneTest, PlaneSceneVertical) {
  // Create a scene that's just a plane.
  // Plane at 1.0 in the positive z direction, pointing in -z.
  constexpr float kPlaneDist = 1.F;

  // Get a camera pointing to this plane from the origin.
  Eigen::Quaternionf rotation =
      Eigen::Quaternionf::FromTwoVectors(Vector3f(0, 0, 1), Vector3f(1, 0, 0));
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation);

  DepthImage depth_frame = this->createPlaneDepthImge(
      primitives::Plane(Vector3f(kPlaneDist, 0, 0), Vector3f(-1, 0, 0)), T_S_C,
      this->sensor());

  // Check all the pixels. Note that we do this differently depending on the
  // sensor type.
  this->assertDepthValid(depth_frame, kPlaneDist);
}

TYPED_TEST(SceneTest, PlaneSceneVerticalOffset) {
  // Create a scene that's just a plane.
  // Plane at 1.0 in the positive z direction, pointing in -z.
  constexpr float kPlaneDist = 1.F;
  constexpr float kTranslation = 1.F;

  // Get a camera pointing to this plane from the origin.
  Eigen::Quaternionf rotation =
      Eigen::Quaternionf::FromTwoVectors(Vector3f(0, 0, 1), Vector3f(1, 0, 0));
  Vector3f translation(-kTranslation, 0, 0);
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation.normalized());
  T_S_C.pretranslate(translation);

  DepthImage depth_frame = this->createPlaneDepthImge(
      primitives::Plane(Vector3f(kPlaneDist, 0, 0), Vector3f(-1, 0, 0)), T_S_C,
      this->sensor());

  // Check all the pixels. Note that we do this differently depending on the
  // sensor type.
  this->assertDepthValid(depth_frame, kPlaneDist + kTranslation);
}

using SceneTestCamera = SceneTest<Camera>;
TEST_F(SceneTestCamera, TypesList) {
  scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(1.0f, 0.0, 0.0), Vector3f(-1, 0, 0)));
  scene_.addPrimitive(
      std::make_unique<primitives::Sphere>(Vector3f(0.0f, 0.0, 0.0), 1.0));
  scene_.addPrimitive(std::make_unique<primitives::Cube>(
      Vector3f(0.0f, 0.0, 0.0), Vector3f(1.0, 1.0, 1.0)));
  scene_.addPrimitive(std::make_unique<primitives::Cylinder>(
      Vector3f(0.0f, 0.0, 0.0), 1.0, 1.0));

  std::vector<primitives::Primitive::Type> type_list =
      scene_.getPrimitiveTypeList();

  EXPECT_EQ(type_list.size(), 4);
  EXPECT_EQ(type_list[0], primitives::Primitive::Type::kPlane);
  EXPECT_EQ(type_list[1], primitives::Primitive::Type::kSphere);
  EXPECT_EQ(type_list[2], primitives::Primitive::Type::kCube);
  EXPECT_EQ(type_list[3], primitives::Primitive::Type::kCylinder);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
