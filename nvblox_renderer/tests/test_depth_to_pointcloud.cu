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
#include <cuda_runtime.h>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/kernels/depth_to_pointcloud.h"
#include "nvblox/sensors/camera.h"

using namespace nvblox;
using namespace nvblox::renderer;

class DepthToPointCloudTest : public ::testing::Test {
 protected:
  void SetUp() override {
    stream_ = std::make_shared<CudaStreamOwning>();

    // Create a simple pinhole camera (640x480, fx=fy=320, cx=320, cy=240)
    width_ = 640;
    height_ = 480;
    fx_ = 320.0f;
    fy_ = 320.0f;
    cx_ = 320.0f;
    cy_ = 240.0f;
    camera_ = Camera(fx_, fy_, cx_, cy_, width_, height_);
  }

  void TearDown() override {
    stream_->synchronize();
    stream_.reset();
  }

  std::shared_ptr<CudaStream> stream_;
  Camera camera_;
  int width_;
  int height_;
  float fx_, fy_, cx_, cy_;
};

// Test that valid depths produce points
TEST_F(DepthToPointCloudTest, ValidDepthsConvert) {
  // Create a small test image (4x4) with uniform valid depth
  const int test_width = 4;
  const int test_height = 4;
  Camera small_cam(fx_, fy_, cx_, cy_, test_width, test_height);

  std::vector<float> h_depth(test_width * test_height, 2.0f);       // All at 2m
  std::vector<uint8_t> h_color(test_width * test_height * 3, 128);  // Gray

  // Allocate device memory
  float* d_depth;
  uint8_t* d_color;
  PointCloudVisualizer::Point* d_points;
  int* d_num_points;

  cudaMalloc(&d_depth, h_depth.size() * sizeof(float));
  cudaMalloc(&d_color, h_color.size() * sizeof(uint8_t));
  cudaMalloc(&d_points,
             test_width * test_height * sizeof(PointCloudVisualizer::Point));
  cudaMalloc(&d_num_points, sizeof(int));

  // Copy input to device
  cudaMemcpy(d_depth, h_depth.data(), h_depth.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_color, h_color.data(), h_color.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  // Run kernel
  const int max_points = test_width * test_height;
  depthToColoredPointCloud(d_depth, d_color, small_cam, small_cam, nullptr,
                           d_points, max_points, d_num_points, 0.1f, 10.0f,
                           *stream_);
  stream_->synchronize();

  // Get point count
  int num_points;
  cudaMemcpy(&num_points, d_num_points, sizeof(int), cudaMemcpyDeviceToHost);

  // All pixels should produce points
  EXPECT_EQ(num_points, test_width * test_height);

  // Cleanup
  cudaFree(d_depth);
  cudaFree(d_color);
  cudaFree(d_points);
  cudaFree(d_num_points);
}

// Test that invalid depths are filtered out
TEST_F(DepthToPointCloudTest, InvalidDepthsFiltered) {
  const int test_width = 4;
  const int test_height = 4;
  Camera small_cam(fx_, fy_, cx_, cy_, test_width, test_height);

  // Create depth image with some invalid values
  std::vector<float> h_depth(test_width * test_height);
  int expected_valid = 0;

  for (int i = 0; i < test_width * test_height; ++i) {
    if (i % 4 == 0) {
      h_depth[i] = 0.0f;  // Invalid: zero depth
    } else if (i % 4 == 1) {
      h_depth[i] = 0.05f;  // Invalid: below min_depth (0.1)
    } else if (i % 4 == 2) {
      h_depth[i] = 15.0f;  // Invalid: above max_depth (10.0)
    } else {
      h_depth[i] = 2.0f;  // Valid
      expected_valid++;
    }
  }

  std::vector<uint8_t> h_color(test_width * test_height * 3, 128);

  // Allocate device memory
  float* d_depth;
  uint8_t* d_color;
  PointCloudVisualizer::Point* d_points;
  int* d_num_points;

  cudaMalloc(&d_depth, h_depth.size() * sizeof(float));
  cudaMalloc(&d_color, h_color.size() * sizeof(uint8_t));
  cudaMalloc(&d_points,
             test_width * test_height * sizeof(PointCloudVisualizer::Point));
  cudaMalloc(&d_num_points, sizeof(int));

  cudaMemcpy(d_depth, h_depth.data(), h_depth.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_color, h_color.data(), h_color.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  // Run with min_depth=0.1, max_depth=10.0
  const int max_points = test_width * test_height;
  depthToColoredPointCloud(d_depth, d_color, small_cam, small_cam, nullptr,
                           d_points, max_points, d_num_points, 0.1f, 10.0f,
                           *stream_);
  stream_->synchronize();

  int num_points;
  cudaMemcpy(&num_points, d_num_points, sizeof(int), cudaMemcpyDeviceToHost);

  // Only valid depths should produce points
  EXPECT_EQ(num_points, expected_valid);

  cudaFree(d_depth);
  cudaFree(d_color);
  cudaFree(d_points);
  cudaFree(d_num_points);
}

// Test that NaN and Inf depths are filtered
TEST_F(DepthToPointCloudTest, HandlesNanAndInf) {
  const int test_width = 4;
  const int test_height = 1;
  Camera small_cam(fx_, fy_, cx_, cy_, test_width, test_height);

  std::vector<float> h_depth = {
      std::numeric_limits<float>::quiet_NaN(),  // Invalid
      std::numeric_limits<float>::infinity(),   // Invalid
      -std::numeric_limits<float>::infinity(),  // Invalid
      2.0f                                      // Valid
  };
  std::vector<uint8_t> h_color(test_width * test_height * 3, 128);

  float* d_depth;
  uint8_t* d_color;
  PointCloudVisualizer::Point* d_points;
  int* d_num_points;

  cudaMalloc(&d_depth, h_depth.size() * sizeof(float));
  cudaMalloc(&d_color, h_color.size() * sizeof(uint8_t));
  cudaMalloc(&d_points,
             test_width * test_height * sizeof(PointCloudVisualizer::Point));
  cudaMalloc(&d_num_points, sizeof(int));

  cudaMemcpy(d_depth, h_depth.data(), h_depth.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_color, h_color.data(), h_color.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  const int max_points = test_width * test_height;
  depthToColoredPointCloud(d_depth, d_color, small_cam, small_cam, nullptr,
                           d_points, max_points, d_num_points, 0.1f, 10.0f,
                           *stream_);
  stream_->synchronize();

  int num_points;
  cudaMemcpy(&num_points, d_num_points, sizeof(int), cudaMemcpyDeviceToHost);

  // Only the last pixel should be valid
  EXPECT_EQ(num_points, 1);

  cudaFree(d_depth);
  cudaFree(d_color);
  cudaFree(d_points);
  cudaFree(d_num_points);
}

// Test that colors are correctly sampled
TEST_F(DepthToPointCloudTest, ColorsMapped) {
  const int test_width = 2;
  const int test_height = 2;
  Camera small_cam(fx_, fy_, cx_, cy_, test_width, test_height);

  // All pixels at same depth
  std::vector<float> h_depth(test_width * test_height, 2.0f);

  // Different color per pixel (RGB)
  std::vector<uint8_t> h_color = {
      255, 0,   0,    // pixel 0: red
      0,   255, 0,    // pixel 1: green
      0,   0,   255,  // pixel 2: blue
      255, 255, 0     // pixel 3: yellow
  };

  float* d_depth;
  uint8_t* d_color;
  PointCloudVisualizer::Point* d_points;
  int* d_num_points;

  cudaMalloc(&d_depth, h_depth.size() * sizeof(float));
  cudaMalloc(&d_color, h_color.size() * sizeof(uint8_t));
  cudaMalloc(&d_points,
             test_width * test_height * sizeof(PointCloudVisualizer::Point));
  cudaMalloc(&d_num_points, sizeof(int));

  cudaMemcpy(d_depth, h_depth.data(), h_depth.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_color, h_color.data(), h_color.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  const int max_points = test_width * test_height;
  depthToColoredPointCloud(d_depth, d_color, small_cam, small_cam, nullptr,
                           d_points, max_points, d_num_points, 0.1f, 10.0f,
                           *stream_);
  stream_->synchronize();

  // Copy output
  std::vector<PointCloudVisualizer::Point> h_points(test_width * test_height);
  cudaMemcpy(h_points.data(), d_points,
             test_width * test_height * sizeof(PointCloudVisualizer::Point),
             cudaMemcpyDeviceToHost);

  int num_points;
  cudaMemcpy(&num_points, d_num_points, sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(num_points, 4);

  // Note: Due to atomicAdd, output order is not guaranteed
  // Check that all expected colors exist in output
  bool found_red = false, found_green = false, found_blue = false,
       found_yellow = false;
  for (int i = 0; i < num_points; ++i) {
    if (h_points[i].r == 255 && h_points[i].g == 0 && h_points[i].b == 0)
      found_red = true;
    if (h_points[i].r == 0 && h_points[i].g == 255 && h_points[i].b == 0)
      found_green = true;
    if (h_points[i].r == 0 && h_points[i].g == 0 && h_points[i].b == 255)
      found_blue = true;
    if (h_points[i].r == 255 && h_points[i].g == 255 && h_points[i].b == 0)
      found_yellow = true;
    // Alpha should always be 255
    EXPECT_EQ(h_points[i].a, 255);
  }

  EXPECT_TRUE(found_red);
  EXPECT_TRUE(found_green);
  EXPECT_TRUE(found_blue);
  EXPECT_TRUE(found_yellow);

  cudaFree(d_depth);
  cudaFree(d_color);
  cudaFree(d_points);
  cudaFree(d_num_points);
}

// Test 3D point positions are computed correctly
TEST_F(DepthToPointCloudTest, PointPositionsCorrect) {
  // Single pixel at image center (cx, cy) with known depth
  const int test_width = 1;
  const int test_height = 1;

  // Camera with center at (0,0)
  float fx = 100.0f, fy = 100.0f, cx = 0.0f, cy = 0.0f;
  Camera single_cam(fx, fy, cx, cy, test_width, test_height);

  float depth_val = 5.0f;
  std::vector<float> h_depth = {depth_val};
  std::vector<uint8_t> h_color = {128, 128, 128};

  float* d_depth;
  uint8_t* d_color;
  PointCloudVisualizer::Point* d_points;
  int* d_num_points;

  cudaMalloc(&d_depth, sizeof(float));
  cudaMalloc(&d_color, 3 * sizeof(uint8_t));
  cudaMalloc(&d_points, sizeof(PointCloudVisualizer::Point));
  cudaMalloc(&d_num_points, sizeof(int));

  cudaMemcpy(d_depth, h_depth.data(), sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_color, h_color.data(), 3 * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  const int max_points = test_width * test_height;
  depthToColoredPointCloud(d_depth, d_color, single_cam, single_cam, nullptr,
                           d_points, max_points, d_num_points, 0.1f, 10.0f,
                           *stream_);
  stream_->synchronize();

  PointCloudVisualizer::Point h_point;
  cudaMemcpy(&h_point, d_points, sizeof(PointCloudVisualizer::Point),
             cudaMemcpyDeviceToHost);

  int num_points;
  cudaMemcpy(&num_points, d_num_points, sizeof(int), cudaMemcpyDeviceToHost);
  EXPECT_EQ(num_points, 1);

  // For pixel at center (0,0) with depth d:
  // x_d = (px - cx) * d / fx = (0 - 0) * 5 / 100 = 0
  // y_d = (py - cy) * d / fy = (0 - 0) * 5 / 100 = 0
  // z_d = d = 5
  // Output has x negated (mirror) and y negated (flip)
  EXPECT_NEAR(h_point.z, depth_val, 0.01f);
  // x and y should be near 0 (negated 0 is still 0)
  EXPECT_NEAR(h_point.x, 0.0f, 0.1f);
  EXPECT_NEAR(h_point.y, 0.0f, 0.1f);

  cudaFree(d_depth);
  cudaFree(d_color);
  cudaFree(d_points);
  cudaFree(d_num_points);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
