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

using namespace nvblox;

constexpr float kFloatEpsilon = 1e-4;

class SceneTest : public ::testing::Test {
 protected:
  SceneTest() : camera_(Camera(fu_, fv_, cu_, cv_, width_, height_)) {}

  void SetUp() override {
    // Make the scene 6x6x3 meters big.
    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-3.0f, -3.0f, 0.0f),
                                           Vector3f(3.0f, 3.0f, 3.0f));
  }

  // A simulation scene.
  primitives::Scene scene_;

  // Camera parameters.
  constexpr static float fu_ = 300;
  constexpr static float fv_ = 300;
  constexpr static int width_ = 640;
  constexpr static int height_ = 480;
  constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
  constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
  Camera camera_;
};

TEST_F(SceneTest, BlankMap) {
  constexpr float max_dist = 1.0;

  // Ensure that we don't get any distances back.
  Vector3f point = Vector3f::Zero();

  float dist = scene_.getSignedDistanceToPoint(point, max_dist);

  EXPECT_NEAR(dist, max_dist, kFloatEpsilon);

  // Ensure that we get a blank image.
  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);
  Transform T_S_C = Transform::Identity();
  scene_.generateDepthImageFromScene(camera_, T_S_C, max_dist, &depth_frame);

  // Check all the pixels.
  for (int lin_idx = 0; lin_idx < depth_frame.numel(); lin_idx++) {
    EXPECT_NEAR(depth_frame(lin_idx), 0.0f, kFloatEpsilon);
  }
}

TEST_F(SceneTest, PlaneScene) {
  constexpr float max_dist = 2.0;

  // Create a scene that's just a plane.
  // Plane at 1.0 in the positive z direction, pointing in -z.
  scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(0.0f, 0.0, 1.0), Vector3f(0, 0, -1)));

  // Get a camera pointing to this plane from the origin.
  Transform T_S_C = Transform::Identity();

  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);
  scene_.generateDepthImageFromScene(camera_, T_S_C, max_dist, &depth_frame);

  // Check all the pixels.
  for (int lin_idx = 0; lin_idx < depth_frame.numel(); lin_idx++) {
    EXPECT_NEAR(depth_frame(lin_idx), 1.0f, kFloatEpsilon);
  }
}

TEST_F(SceneTest, PlaneSceneVertical) {
  constexpr float max_dist = 2.0;

  // Create a scene that's just a plane.
  // Plane at 1.0 in the positive x direction, pointing in -x.
  scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(1.0f, 0.0, 0.0), Vector3f(-1, 0, 0)));

  // Get a camera pointing to this plane from the origin.
  Eigen::Quaternionf rotation =
      Eigen::Quaternionf::FromTwoVectors(Vector3f(0, 0, 1), Vector3f(1, 0, 0));

  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation);

  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);
  scene_.generateDepthImageFromScene(camera_, T_S_C, max_dist, &depth_frame);
  io::writeToPng("test_plane_scene_vertical.png", depth_frame);

  // Check all the pixels.
  for (int lin_idx = 0; lin_idx < depth_frame.numel(); lin_idx++) {
    EXPECT_NEAR(depth_frame(lin_idx), 1.0f, kFloatEpsilon);
  }
}

TEST_F(SceneTest, PlaneSceneVerticalOffset) {
  constexpr float max_dist = 4.0;

  // Create a scene that's just a plane.
  // Plane at 1.0 in the positive x direction, pointing in -x.
  scene_.addPrimitive(std::make_unique<primitives::Plane>(
      Vector3f(1.0f, 0.0, 0.0), Vector3f(-1, 0, 0)));

  // Get a camera pointing to this plane from (-1, 0, 0).
  Eigen::Quaternionf rotation =
      Eigen::Quaternionf::FromTwoVectors(Vector3f(0, 0, 1), Vector3f(1, 0, 0));
  Vector3f translation(-1, 0, 0);
  Transform T_S_C = Transform::Identity();
  T_S_C.prerotate(rotation.normalized());
  T_S_C.pretranslate(translation);

  DepthImage depth_frame(camera_.height(), camera_.width(),
                         MemoryType::kUnified);
  scene_.generateDepthImageFromScene(camera_, T_S_C, max_dist, &depth_frame);

  // Check all the pixels.
  for (int lin_idx = 0; lin_idx < depth_frame.numel(); lin_idx++) {
    EXPECT_NEAR(depth_frame(lin_idx), 2.0f, kFloatEpsilon);
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
