/*
Copyright 2023 NVIDIA CORPORATION

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

#include "nvblox/datasets/3dmatch.h"
#include "nvblox/io/image_io.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/npp_image_operations.h"

using namespace nvblox;

// Flag to write test data. Can be useful during troubleshooting.
constexpr bool writeDebugOutput = false;

class NppImageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Load a test depth frame to unified memory
    const std::string base_path = "./data/3dmatch/";
    constexpr int seq_id = 1;
    constexpr bool kMultithreaded = false;
    depth_loader_ptr_ = datasets::threedmatch::internal::createDepthImageLoader(
        base_path, seq_id, kMultithreaded);

    // Print some debug information
    image::printNPPVersionInfo();
    // NPP stream context.
    stream_context_ = image::getNppStreamContext(cuda_stream_);
  }

  bool getDepthFrame(DepthImage* depth_image_ptr) {
    return depth_loader_ptr_->getNextImage(depth_image_ptr);
  }

  std::unique_ptr<datasets::ImageLoader<DepthImage>> depth_loader_ptr_;

  // Stream
  CudaStreamOwning cuda_stream_;
  NppStreamContext stream_context_;
};

class ParameterizedNppImageTest
    : public NppImageTest,
      public ::testing::WithParamInterface<MemoryType> {
 protected:
};

TEST_P(ParameterizedNppImageTest, NppThresholdDepthImage) {
  // This test runs for multiple memory types.
  const MemoryType memory_type = GetParam();
  // Load the test image
  DepthImage depth_frame(memory_type);
  getDepthFrame(&depth_frame);
  // Invalid depth threshold
  constexpr float kInvalidDepthThreshold = 1e-2;
  // Allocate the output space
  MonoImage output(depth_frame.rows(), depth_frame.cols(),
                   MemoryType::kUnified);
  // Call the comparison (ROI is the whole image)
  image::getInvalidDepthMaskAsync(depth_frame, stream_context_, &output,
                                  kInvalidDepthThreshold);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
  // Check the mask on CPU
  for (int row_idx = 0; row_idx < depth_frame.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < depth_frame.rows(); col_idx++) {
      const float depth_value = depth_frame(row_idx, col_idx);
      const uint8_t mask_value = output(row_idx, col_idx);
      if (depth_value < kInvalidDepthThreshold) {
        EXPECT_EQ(mask_value, NPP_MAX_8U);
      } else {
        EXPECT_EQ(mask_value, 0);
      }
    }
  }
  // Check the result
  if (writeDebugOutput) {
    io::writeToPng("./invalid_mask.png", output);
  }
}

TEST_P(ParameterizedNppImageTest, NppDilateMask) {
  // This test runs for multiple memory types.
  const MemoryType memory_type = GetParam();
  // Load the test image
  DepthImage depth_frame(memory_type);
  getDepthFrame(&depth_frame);
  // Allocate the output space
  MonoImage mask(depth_frame.rows(), depth_frame.cols(), MemoryType::kUnified);
  // Call the comparison (ROI is the whole image)
  image::getInvalidDepthMaskAsync(depth_frame, stream_context_, &mask);
  // Dilate the mask
  MonoImage mask_dilated(depth_frame.rows(), depth_frame.cols(), memory_type);
  image::dilateMask3x3Async(mask, stream_context_, &mask_dilated);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  MonoImage mask_dilated_test(depth_frame.rows(), depth_frame.cols(),
                              memory_type);

  // Check the mask on CPU
  for (int row_idx = 1; row_idx < depth_frame.rows() - 1; row_idx++) {
    for (int col_idx = 1; col_idx < depth_frame.cols() - 1; col_idx++) {
      // Manual dilation
      uint8_t dilated_value = 0;
      for (int i = -1; i <= 1; i++) {
        for (int j = -1; j <= 1; j++) {
          if (mask(row_idx + i, col_idx + j) > 0) {
            dilated_value = NPP_MAX_8U;
          }
        }
      }
      mask_dilated_test(row_idx, col_idx) = dilated_value;
      EXPECT_EQ(dilated_value, mask_dilated(row_idx, col_idx));
    }
  }
  // Write debug
  if (writeDebugOutput) {
    io::writeToPng("./invalid_mask_dilated.png", mask_dilated);
    io::writeToPng("./invalid_mask_dilated_test.png", mask_dilated_test);
  }
}

TEST_P(ParameterizedNppImageTest, NppDilateMaskBorder) {
  // This test runs for multiple memory types.
  const MemoryType memory_type = GetParam();
  // Dummy image
  constexpr int kRows = 3;
  constexpr int kCols = 3;
  MonoImage mask(kRows, kCols, memory_type);

  // Center pixel 255
  mask.setZero();
  mask(1, 1) = NPP_MAX_8U;

  // Dilate
  MonoImage mask_dilated(mask.rows(), mask.cols(), memory_type);
  image::dilateMask3x3Async(mask, stream_context_, &mask_dilated);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  // Check the mask on CPU
  for (int row_idx = 0; row_idx < mask.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < mask.cols(); col_idx++) {
      EXPECT_EQ(mask_dilated(row_idx, col_idx), NPP_MAX_8U);
    }
  }

  // Upper left 255
  mask.setZero();
  mask(0, 0) = NPP_MAX_8U;

  // Dilate again
  image::dilateMask3x3Async(mask, stream_context_, &mask_dilated);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  EXPECT_EQ(mask_dilated(0, 0), NPP_MAX_8U);
  EXPECT_EQ(mask_dilated(0, 1), NPP_MAX_8U);
  EXPECT_EQ(mask_dilated(1, 0), NPP_MAX_8U);
  EXPECT_EQ(mask_dilated(1, 1), NPP_MAX_8U);

  EXPECT_EQ(mask_dilated(0, 2), 0);
  EXPECT_EQ(mask_dilated(2, 0), 0);
  EXPECT_EQ(mask_dilated(1, 2), 0);
  EXPECT_EQ(mask_dilated(2, 1), 0);
  EXPECT_EQ(mask_dilated(2, 2), 0);
}

TEST_P(ParameterizedNppImageTest, NppSetMasked) {
  // This test runs for multiple memory types.
  const MemoryType memory_type = GetParam();
  // Load the test image
  DepthImage depth_frame(memory_type);
  getDepthFrame(&depth_frame);
  // Create a mask where every second pixel is true.
  MonoImage mask(depth_frame.rows(), depth_frame.cols(), memory_type);
  mask.setZero();
  for (int row_idx = 0; row_idx < depth_frame.rows(); row_idx += 2) {
    for (int col_idx = 0; col_idx < depth_frame.cols(); col_idx += 2) {
      mask(row_idx, col_idx) = NPP_MAX_8U;
    }
  }
  // Create a copy of the depth image
  DepthImage depth_image_original(memory_type);
  depth_image_original.copyFrom(depth_frame);
  // Do the masked set
  constexpr float kInvalidDepthValue = 0.f;
  image::maskedSetAsync(mask, kInvalidDepthValue, stream_context_,
                        &depth_frame);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  // Check the masked set worked
  for (int row_idx = 0; row_idx < depth_frame.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < depth_frame.cols(); col_idx++) {
      // Every second pixel is zeroed
      constexpr float kEps = 1e-6;
      if ((row_idx % 2 == 0) && (col_idx % 2 == 0)) {
        EXPECT_NEAR(depth_frame(row_idx, col_idx), 0.f, kEps);
      }
      // Every other pixel is unaffected
      else {
        EXPECT_NEAR(depth_frame(row_idx, col_idx),
                    depth_image_original(row_idx, col_idx), kEps);
      }
    }
  }

  // Write debug
  if (writeDebugOutput) {
    io::writeToPng("./masked_set_depth_frame.png", depth_frame);
  }
}

TEST_P(ParameterizedNppImageTest, setGreaterThanThresholdToValue) {
  MonoImage image(100, 100, MemoryType::kHost);
  image.setZero();

  constexpr uint8_t kThreshold = 10;
  constexpr uint8_t kSetToValue = 200;

  // left half of image: below threshold
  // right half of image: above theshold
  for (int y = 0; y < image.rows(); ++y) {
    for (int x = 0; x < image.cols(); ++x) {
      if (x < image.cols() / 2) {
        image(y, x) = kThreshold - 1;
      } else {
        image(y, x) = kThreshold + 1;
      }
    }
  }

  // apply thresholding
  MonoImage thresholded_image(image.rows(), image.cols(), MemoryType::kHost);
  image::setGreaterThanThresholdToValue(image, kThreshold, kSetToValue,
                                        stream_context_, &thresholded_image);
  cudaStreamSynchronize(stream_context_.hStream);

  // check that all colored values changed
  for (int y = 0; y < image.rows(); ++y) {
    for (int x = 0; x < image.cols(); ++x) {
      if (x < image.cols() / 2) {
        EXPECT_EQ(thresholded_image(y, x), kThreshold - 1);
      } else {
        EXPECT_EQ(thresholded_image(y, x), kSetToValue);
      }
    }
  }
}

INSTANTIATE_TEST_CASE_P(ParameterizedNppImageTests, ParameterizedNppImageTest,
                        ::testing::Values(MemoryType::kUnified,
                                          MemoryType::kHost));

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
