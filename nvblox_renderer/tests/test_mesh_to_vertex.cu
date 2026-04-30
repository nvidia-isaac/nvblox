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
#include "nvblox/renderer/kernels/mesh_to_vertex.h"

using namespace nvblox;
using namespace nvblox::renderer;

class MeshToVertexTest : public ::testing::Test {
 protected:
  void SetUp() override { stream_ = std::make_shared<CudaStreamOwning>(); }

  void TearDown() override {
    stream_->synchronize();
    stream_.reset();
  }

  std::shared_ptr<CudaStream> stream_;
};

// Test basic interleaving correctness with small data
TEST_F(MeshToVertexTest, InterleavesCorrectly) {
  const size_t num_vertices = 4;

  // Input positions (x, y, z per vertex)
  std::vector<float> h_positions = {
      1.0f,  2.0f,  3.0f,  // vertex 0
      4.0f,  5.0f,  6.0f,  // vertex 1
      7.0f,  8.0f,  9.0f,  // vertex 2
      10.0f, 11.0f, 12.0f  // vertex 3
  };

  // Input colors (r, g, b per vertex)
  std::vector<uint8_t> h_colors = {
      255, 0,   0,    // vertex 0: red
      0,   255, 0,    // vertex 1: green
      0,   0,   255,  // vertex 2: blue
      128, 128, 128   // vertex 3: gray
  };

  // Allocate device memory
  float* d_positions;
  uint8_t* d_colors;
  MeshVertex* d_output;

  cudaMalloc(&d_positions, h_positions.size() * sizeof(float));
  cudaMalloc(&d_colors, h_colors.size() * sizeof(uint8_t));
  cudaMalloc(&d_output, num_vertices * sizeof(MeshVertex));

  // Copy input to device
  cudaMemcpy(d_positions, h_positions.data(),
             h_positions.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_colors, h_colors.data(), h_colors.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  // Run kernel
  interleaveMeshVertexData(d_positions, d_colors, d_output, num_vertices,
                           *stream_);
  stream_->synchronize();

  // Copy output back to host
  std::vector<MeshVertex> h_output(num_vertices);
  cudaMemcpy(h_output.data(), d_output, num_vertices * sizeof(MeshVertex),
             cudaMemcpyDeviceToHost);

  // Verify vertex 0
  EXPECT_FLOAT_EQ(h_output[0].x, 1.0f);
  EXPECT_FLOAT_EQ(h_output[0].y, 2.0f);
  EXPECT_FLOAT_EQ(h_output[0].z, 3.0f);
  EXPECT_EQ(h_output[0].r, 255);
  EXPECT_EQ(h_output[0].g, 0);
  EXPECT_EQ(h_output[0].b, 0);
  EXPECT_EQ(h_output[0].a, 255);

  // Verify vertex 1
  EXPECT_FLOAT_EQ(h_output[1].x, 4.0f);
  EXPECT_FLOAT_EQ(h_output[1].y, 5.0f);
  EXPECT_FLOAT_EQ(h_output[1].z, 6.0f);
  EXPECT_EQ(h_output[1].r, 0);
  EXPECT_EQ(h_output[1].g, 255);
  EXPECT_EQ(h_output[1].b, 0);
  EXPECT_EQ(h_output[1].a, 255);

  // Verify vertex 2
  EXPECT_FLOAT_EQ(h_output[2].x, 7.0f);
  EXPECT_FLOAT_EQ(h_output[2].y, 8.0f);
  EXPECT_FLOAT_EQ(h_output[2].z, 9.0f);
  EXPECT_EQ(h_output[2].r, 0);
  EXPECT_EQ(h_output[2].g, 0);
  EXPECT_EQ(h_output[2].b, 255);
  EXPECT_EQ(h_output[2].a, 255);

  // Verify vertex 3
  EXPECT_FLOAT_EQ(h_output[3].x, 10.0f);
  EXPECT_FLOAT_EQ(h_output[3].y, 11.0f);
  EXPECT_FLOAT_EQ(h_output[3].z, 12.0f);
  EXPECT_EQ(h_output[3].r, 128);
  EXPECT_EQ(h_output[3].g, 128);
  EXPECT_EQ(h_output[3].b, 128);
  EXPECT_EQ(h_output[3].a, 255);

  // Cleanup
  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_output);
}

// Test with realistic mesh size (100K+ vertices)
TEST_F(MeshToVertexTest, HandlesLargeCount) {
  const size_t num_vertices = 100000;

  // Generate test data
  std::vector<float> h_positions(num_vertices * 3);
  std::vector<uint8_t> h_colors(num_vertices * 3);

  for (size_t i = 0; i < num_vertices; ++i) {
    h_positions[i * 3 + 0] = static_cast<float>(i);
    h_positions[i * 3 + 1] = static_cast<float>(i + 1);
    h_positions[i * 3 + 2] = static_cast<float>(i + 2);
    h_colors[i * 3 + 0] = static_cast<uint8_t>(i % 256);
    h_colors[i * 3 + 1] = static_cast<uint8_t>((i + 1) % 256);
    h_colors[i * 3 + 2] = static_cast<uint8_t>((i + 2) % 256);
  }

  // Allocate device memory
  float* d_positions;
  uint8_t* d_colors;
  MeshVertex* d_output;

  cudaMalloc(&d_positions, h_positions.size() * sizeof(float));
  cudaMalloc(&d_colors, h_colors.size() * sizeof(uint8_t));
  cudaMalloc(&d_output, num_vertices * sizeof(MeshVertex));

  // Copy input to device
  cudaMemcpy(d_positions, h_positions.data(),
             h_positions.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_colors, h_colors.data(), h_colors.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  // Run kernel
  interleaveMeshVertexData(d_positions, d_colors, d_output, num_vertices,
                           *stream_);
  stream_->synchronize();

  // Copy output back to host
  std::vector<MeshVertex> h_output(num_vertices);
  cudaMemcpy(h_output.data(), d_output, num_vertices * sizeof(MeshVertex),
             cudaMemcpyDeviceToHost);

  // Spot check some vertices
  size_t test_indices[] = {0, 1000, 50000, 99999};
  for (size_t i : test_indices) {
    EXPECT_FLOAT_EQ(h_output[i].x, static_cast<float>(i));
    EXPECT_FLOAT_EQ(h_output[i].y, static_cast<float>(i + 1));
    EXPECT_FLOAT_EQ(h_output[i].z, static_cast<float>(i + 2));
    EXPECT_EQ(h_output[i].r, static_cast<uint8_t>(i % 256));
    EXPECT_EQ(h_output[i].g, static_cast<uint8_t>((i + 1) % 256));
    EXPECT_EQ(h_output[i].b, static_cast<uint8_t>((i + 2) % 256));
    EXPECT_EQ(h_output[i].a, 255);
  }

  // Cleanup
  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_output);
}

// Test that values are preserved exactly (no rounding/truncation)
TEST_F(MeshToVertexTest, PreservesValues) {
  const size_t num_vertices = 3;

  // Use specific float values that could be affected by rounding
  std::vector<float> h_positions = {
      0.123456789f, -0.987654321f, 1e-7f,       // vertex 0
      1e7f,         -1e7f,         0.0f,        // vertex 1
      3.14159265f,  2.71828182f,   1.41421356f  // vertex 2
  };

  std::vector<uint8_t> h_colors = {
      0,   1,   254,  // edge values
      127, 128, 129,  // middle values
      255, 0,   127   // mixed
  };

  // Allocate device memory
  float* d_positions;
  uint8_t* d_colors;
  MeshVertex* d_output;

  cudaMalloc(&d_positions, h_positions.size() * sizeof(float));
  cudaMalloc(&d_colors, h_colors.size() * sizeof(uint8_t));
  cudaMalloc(&d_output, num_vertices * sizeof(MeshVertex));

  // Copy input to device
  cudaMemcpy(d_positions, h_positions.data(),
             h_positions.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_colors, h_colors.data(), h_colors.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  // Run kernel
  interleaveMeshVertexData(d_positions, d_colors, d_output, num_vertices,
                           *stream_);
  stream_->synchronize();

  // Copy output back to host
  std::vector<MeshVertex> h_output(num_vertices);
  cudaMemcpy(h_output.data(), d_output, num_vertices * sizeof(MeshVertex),
             cudaMemcpyDeviceToHost);

  // Verify exact preservation
  for (size_t i = 0; i < num_vertices; ++i) {
    EXPECT_FLOAT_EQ(h_output[i].x, h_positions[i * 3 + 0]);
    EXPECT_FLOAT_EQ(h_output[i].y, h_positions[i * 3 + 1]);
    EXPECT_FLOAT_EQ(h_output[i].z, h_positions[i * 3 + 2]);
    EXPECT_EQ(h_output[i].r, h_colors[i * 3 + 0]);
    EXPECT_EQ(h_output[i].g, h_colors[i * 3 + 1]);
    EXPECT_EQ(h_output[i].b, h_colors[i * 3 + 2]);
  }

  // Cleanup
  cudaFree(d_positions);
  cudaFree(d_colors);
  cudaFree(d_output);
}

// Test empty input (edge case)
TEST_F(MeshToVertexTest, HandlesEmptyInput) {
  // Should not crash with zero vertices and should not produce CUDA errors
  interleaveMeshVertexData(nullptr, nullptr, nullptr, 0, *stream_);
  stream_->synchronize();

  // Verify no CUDA errors were generated
  cudaError_t err = cudaGetLastError();
  EXPECT_EQ(err, cudaSuccess) << "Empty input should not produce CUDA errors: "
                              << cudaGetErrorString(err);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
