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

#include "nvblox/renderer/utils/view_camera.h"

using namespace nvblox::renderer;

constexpr float kEpsilon = 1e-5f;

class ViewCameraTest : public ::testing::Test {
 protected:
  ViewCamera camera_;
};

// Test that initial state has valid position and matrices
TEST_F(ViewCameraTest, InitialState) {
  // Default distance is 3.0, target is origin
  EXPECT_FLOAT_EQ(camera_.distance(), 3.0f);
  EXPECT_FLOAT_EQ(camera_.target().x(), 0.0f);
  EXPECT_FLOAT_EQ(camera_.target().y(), 0.0f);
  EXPECT_FLOAT_EQ(camera_.target().z(), 0.0f);

  // Position should be at distance from target
  Eigen::Vector3f pos = camera_.position();
  float dist = (pos - camera_.target()).norm();
  EXPECT_NEAR(dist, camera_.distance(), kEpsilon);

  // View matrix should be valid (non-zero determinant)
  Eigen::Matrix4f view = camera_.viewMatrix();
  float det = view.determinant();
  EXPECT_NE(det, 0.0f);
}

// Test that arcball rotation preserves distance from target
TEST_F(ViewCameraTest, RotationPreservesDistance) {
  float initial_dist = camera_.distance();

  // Perform multiple rotations
  camera_.rotate(0.5f, 0.3f);
  EXPECT_NEAR((camera_.position() - camera_.target()).norm(), initial_dist,
              kEpsilon);

  camera_.rotate(-0.2f, 0.7f);
  EXPECT_NEAR((camera_.position() - camera_.target()).norm(), initial_dist,
              kEpsilon);

  camera_.rotate(1.5f, -0.5f);
  EXPECT_NEAR((camera_.position() - camera_.target()).norm(), initial_dist,
              kEpsilon);
}

// Test that view matrix has orthonormal rotation component
TEST_F(ViewCameraTest, ViewMatrixOrthonormal) {
  camera_.rotate(0.3f, 0.5f);  // Apply some rotation

  Eigen::Matrix4f view = camera_.viewMatrix();

  // Extract 3x3 rotation part (rows of the upper-left 3x3)
  Eigen::Vector3f right(view(0, 0), view(0, 1), view(0, 2));
  Eigen::Vector3f up(view(1, 0), view(1, 1), view(1, 2));
  Eigen::Vector3f forward(view(2, 0), view(2, 1), view(2, 2));

  // Check orthogonality (dot products should be 0)
  EXPECT_NEAR(right.dot(up), 0.0f, kEpsilon);
  EXPECT_NEAR(right.dot(forward), 0.0f, kEpsilon);
  EXPECT_NEAR(up.dot(forward), 0.0f, kEpsilon);

  // Check unit length
  EXPECT_NEAR(right.norm(), 1.0f, kEpsilon);
  EXPECT_NEAR(up.norm(), 1.0f, kEpsilon);
  EXPECT_NEAR(forward.norm(), 1.0f, kEpsilon);
}

// Test that projection matrix has Vulkan Y-flip applied
TEST_F(ViewCameraTest, ProjectionFlipsY) {
  Eigen::Matrix4f proj = camera_.projMatrix();

  // Vulkan convention: Y is flipped (proj(1,1) is negative)
  // Standard OpenGL would have positive proj(1,1)
  EXPECT_LT(proj(1, 1), 0.0f);
}

// Test that zoom cannot go below minimum distance
TEST_F(ViewCameraTest, ZoomClampsMinDistance) {
  // Zoom in a lot (positive delta = zoom in)
  for (int i = 0; i < 100; ++i) {
    camera_.zoom(10.0f);
  }

  // Distance should be clamped to kMinDistance (0.1f)
  EXPECT_GE(camera_.distance(), 0.1f);

  // Position should still be at distance from target
  float dist = (camera_.position() - camera_.target()).norm();
  EXPECT_NEAR(dist, camera_.distance(), kEpsilon);
}

// Test that pan moves the target correctly
TEST_F(ViewCameraTest, PanMovesTarget) {
  Eigen::Vector3f initial_target = camera_.target();

  camera_.pan(100.0f, 0.0f);  // Pan right
  Eigen::Vector3f after_pan = camera_.target();

  // Target should have moved
  EXPECT_NE(after_pan.x(), initial_target.x());

  // Camera should still be at same distance from new target
  float dist = (camera_.position() - camera_.target()).norm();
  EXPECT_NEAR(dist, camera_.distance(), kEpsilon);
}

// Test that reset restores default state
TEST_F(ViewCameraTest, ResetRestoresDefaults) {
  // Modify camera state
  camera_.setTarget(5.0f, 3.0f, -2.0f);
  camera_.setDistance(10.0f);
  camera_.rotate(1.0f, 0.5f);

  // Reset
  camera_.reset();

  // Check defaults are restored
  EXPECT_FLOAT_EQ(camera_.distance(), 3.0f);
  EXPECT_FLOAT_EQ(camera_.target().x(), 0.0f);
  EXPECT_FLOAT_EQ(camera_.target().y(), 0.0f);
  EXPECT_FLOAT_EQ(camera_.target().z(), 0.0f);

  // Rotation should be identity quaternion
  Eigen::Quaternionf rot = camera_.rotation();
  EXPECT_NEAR(rot.w(), 1.0f, kEpsilon);
  EXPECT_NEAR(rot.x(), 0.0f, kEpsilon);
  EXPECT_NEAR(rot.y(), 0.0f, kEpsilon);
  EXPECT_NEAR(rot.z(), 0.0f, kEpsilon);
}

// Test setOrbitAngles converts to quaternion correctly
TEST_F(ViewCameraTest, SetOrbitAnglesWorks) {
  camera_.setOrbitAngles(0.0f, 0.0f);  // Looking along +Z

  // After setting angles, position should be updated
  Eigen::Vector3f pos = camera_.position();
  float dist = (pos - camera_.target()).norm();
  EXPECT_NEAR(dist, camera_.distance(), kEpsilon);
}

// Test view-projection matrix is product of view and projection
TEST_F(ViewCameraTest, ViewProjMatrixIsProduct) {
  camera_.rotate(0.2f, 0.3f);

  Eigen::Matrix4f view = camera_.viewMatrix();
  Eigen::Matrix4f proj = camera_.projMatrix();
  Eigen::Matrix4f view_proj = camera_.viewProjMatrix();

  Eigen::Matrix4f expected = proj * view;

  // Compare matrices element by element
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NEAR(view_proj(i, j), expected(i, j), kEpsilon);
    }
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
