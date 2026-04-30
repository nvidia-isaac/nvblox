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

#include <iostream>

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/datasets/image_loader.h"
#include "nvblox/io/image_io.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

constexpr int kWidth = 1024;
constexpr int kHeight = 1024;
constexpr float kVoxelSize = 0.01;

// Path to the base directory of the Orbec dataset.
std::string base_path() { return std::string("./data/orbbec/"); }
// Path to the camera calibration file.
std::string calib_path() {
  return std::string(base_path() + "camera-intrinsics.txt");
}

// Reconstruct the scene using the Orbec dataset.
void reconstructOrbbec(const Camera& camera, Mapper* mapper) {
  // Read depth frames
  constexpr int kNumFrames = 1;
  for (int i = 0; i < kNumFrames; i++) {
    DepthImage depth_frame(MemoryType::kHost);
    EXPECT_TRUE(io::readFromPng(
        base_path() + "seq-01/frame-00000" + std::to_string(i) + ".depth.png",
        &depth_frame));

    ASSERT_EQ(depth_frame.width(), kWidth);
    ASSERT_EQ(depth_frame.height(), kHeight);

    // Integrate depth frames. Dataset is static, so we can use the same
    // camera transform for all frames.
    mapper->integrateDepth(depth_frame, Transform::Identity(), camera);
  }
  mapper->updateColorMesh();
  mapper->serializeSelectedLayers(LayerType::kColorMesh, -1.0f);
}

// Count the number of vertices that are considered to be part of the ceiling.
// This is done by checking if the y-coordinate is within a certain margin of
// the expected ceiling y-coordinate.
int numCeilingVertices(
    std::shared_ptr<SerializedColorMeshLayer> serialized_color_mesh) {
  // Y-coordinate of expected ceiling in camera frame (manually measured from
  // reconstructed mesh)
  constexpr float kExpectedCeilingY = -2.3f;

  // Allowed margin for a point to be considered a ceiling vertex
  constexpr float kMargin = 0.1f;

  int num_ceiling_vertices = 0;
  for (size_t i = 0; i < serialized_color_mesh->vertices.size(); i++) {
    float vertex_y = serialized_color_mesh->vertices[i].y();
    if (vertex_y > kExpectedCeilingY - kMargin &&
        vertex_y < kExpectedCeilingY + kMargin) {
      ++num_ceiling_vertices;
    }
  }
  return num_ceiling_vertices;
}

// Test that our test scene can be reconstructed.
TEST(TestOrbbec, ReconstructWithCorrectDistortion) {
  // Read camera calibration and reconstruct the scene
  Eigen::Matrix3f camera_intrinsic_matrix;
  RadialTangentialDistortionParams distortion_params;
  EXPECT_TRUE(datasets::threedmatch::internal::parseCameraFromFile(
      calib_path(), &camera_intrinsic_matrix, &distortion_params));
  Camera camera = Camera::fromIntrinsicsMatrix(camera_intrinsic_matrix, kWidth,
                                               kHeight, distortion_params);

  Mapper mapper(kVoxelSize, MemoryType::kHost, ProjectiveLayerType::kTsdf);
  reconstructOrbbec(camera, &mapper);

  // Check that we managed to reconstruct an expected number of ceiling
  // vertices.
  constexpr int kExpectedNumCeilingVertices = 50000;
  EXPECT_GE(numCeilingVertices(mapper.serializedColorMeshLayer()),
            kExpectedNumCeilingVertices);
}

// Test that the scene cannot be reconstruction if the distortion parameters are
// removed from the camera model.
TEST(TestOrbbec, ReconstructWithNoDistortion) {
  // Read camera calibration and reconstruct the scene
  Eigen::Matrix3f camera_intrinsic_matrix;
  RadialTangentialDistortionParams distortion_params;
  EXPECT_TRUE(datasets::threedmatch::internal::parseCameraFromFile(
      calib_path(), &camera_intrinsic_matrix, &distortion_params));
  Camera camera = Camera::fromIntrinsicsMatrix(camera_intrinsic_matrix, kWidth,
                                               kHeight, distortion_params);

  // Reconstruct the scene with a camera where distortion is omitted.
  Camera camera_no_distortion =
      Camera::fromIntrinsicsMatrix(camera_intrinsic_matrix, kWidth, kHeight);
  Mapper mapper_no_distortion(kVoxelSize, MemoryType::kHost,
                              ProjectiveLayerType::kTsdf);
  reconstructOrbbec(camera_no_distortion, &mapper_no_distortion);

  // Since we're not modelling camera distortion here, the ceiling will be
  // curved and displaced. We do not expect any correctly reconstructed points.
  EXPECT_EQ(numCeilingVertices(mapper_no_distortion.serializedColorMeshLayer()),
            0);
}
}  // namespace nvblox

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
