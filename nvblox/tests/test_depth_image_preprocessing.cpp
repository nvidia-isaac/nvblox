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
#include "nvblox/sensors/depth_preprocessing.h"
#include "nvblox/sensors/image.h"

using namespace nvblox;

// Flag to write test data. Can be useful during troubleshooting.
constexpr bool writeDebugOutput = false;

class DepthImagePreprocessing : public ::testing::Test {
 protected:
  void SetUp() override {
    // Load a test depth frame to unified memory
    const std::string base_path = "./data/3dmatch/";
    constexpr int seq_id = 1;
    auto depth_loader_ptr =
        datasets::threedmatch::internal::createDepthImageLoader(base_path,
                                                                seq_id, false);
    depth_loader_ptr->getNextImage(&depth_frame_);
    CHECK(depth_frame_.memory_type() == MemoryType::kUnified);

    cuda_stream_ = std::make_shared<CudaStreamOwning>();
    depth_preprocessor_ptr_ = std::make_shared<DepthPreprocessor>(cuda_stream_);
  }

  std::shared_ptr<DepthPreprocessor> depth_preprocessor_ptr_;

  // Stream
  std::shared_ptr<CudaStream> cuda_stream_;

  // Image
  DepthImage depth_frame_{MemoryType::kUnified};
};

TEST_F(DepthImagePreprocessing, GettersAndSetters) {
  constexpr float kEps = 1e-6;
  depth_preprocessor_ptr_->invalid_depth_threshold(1.0f);
  EXPECT_NEAR(depth_preprocessor_ptr_->invalid_depth_threshold(), 1.0f, kEps);
  depth_preprocessor_ptr_->invalid_depth_value(2.0f);
  EXPECT_NEAR(depth_preprocessor_ptr_->invalid_depth_value(), 2.0f, kEps);
}

TEST_F(DepthImagePreprocessing, NppThresholdDepthImage) {
  DepthImage depth_image_dilated(MemoryType::kUnified);
  depth_image_dilated.copyFromAsync(depth_frame_, *cuda_stream_);
  depth_preprocessor_ptr_->dilateInvalidRegionsAsync(1, &depth_image_dilated);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  // Check manually
  for (int row_idx = 1; row_idx < depth_frame_.rows() - 1; row_idx++) {
    for (int col_idx = 1; col_idx < depth_frame_.cols() - 1; col_idx++) {
      if (depth_frame_(row_idx, col_idx) <
          depth_preprocessor_ptr_->invalid_depth_threshold()) {
        for (int i = -1; i <= 1; i++) {
          for (int j = -1; j <= 1; j++) {
            constexpr float kEps = 1e-6;
            EXPECT_NEAR(depth_image_dilated(row_idx + i, col_idx + j),
                        depth_preprocessor_ptr_->invalid_depth_value(), kEps);
          }
        }
      }
    }
  }

  if (writeDebugOutput) {
    io::writeToPng("./preprocessing_before.png", depth_frame_);
    io::writeToPng("./preprocessing_after.png", depth_image_dilated);
  }
}

void setImageConstantOnCPU(const float value, DepthImage* depth_image_ptr) {
  for (int i = 0; i < depth_image_ptr->numel(); i++) {
    (*depth_image_ptr)(i) = value;
  }
}

float getImageSumOnCPU(const DepthImage& depth_image) {
  float sum = 0.f;
  for (int i = 0; i < depth_image.numel(); i++) {
    sum += depth_image(i);
  }
  return sum;
}

void printImageToConsole(const DepthImage& depth_image) {
  for (int row_idx = 0; row_idx < depth_image.rows(); row_idx++) {
    for (int col_idx = 0; col_idx < depth_image.cols(); col_idx++) {
      std::cout << depth_image(row_idx, col_idx);
    }
    std::cout << std::endl;
  }
}

TEST_F(DepthImagePreprocessing, DilationNumberTests) {
  // Set up dummy image with invalid pixel in the center
  constexpr int kDummyImageSize = 9;
  DepthImage depth_image(kDummyImageSize, kDummyImageSize,
                         MemoryType::kUnified);
  setImageConstantOnCPU(1.f, &depth_image);
  constexpr int kCenterPixelIndex = (kDummyImageSize - 1) / 2;
  depth_image(kCenterPixelIndex, kCenterPixelIndex) = 0.f;

  // Calculate the expected image sum
  auto get_expected_image_sum = [&](const int num_dilations) -> int {
    const int dilated_region_size = 1 + 2 * num_dilations;
    CHECK_GE(kDummyImageSize, dilated_region_size);
    return (kDummyImageSize * kDummyImageSize) -
           (dilated_region_size * dilated_region_size);
  };

  // 1 dilation
  DepthImage depth_image_1_dilation{MemoryType::kUnified};
  depth_image_1_dilation.copyFromAsync(depth_image, *cuda_stream_);
  depth_preprocessor_ptr_->dilateInvalidRegionsAsync(1,
                                                     &depth_image_1_dilation);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
  EXPECT_EQ(getImageSumOnCPU(depth_image_1_dilation),
            get_expected_image_sum(1));

  // 2 dilation
  DepthImage depth_image_2_dilation{MemoryType::kUnified};
  depth_image_2_dilation.copyFromAsync(depth_image, *cuda_stream_);
  depth_preprocessor_ptr_->dilateInvalidRegionsAsync(2,
                                                     &depth_image_2_dilation);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
  EXPECT_EQ(getImageSumOnCPU(depth_image_2_dilation),
            get_expected_image_sum(2));

  // 3 dilation
  DepthImage depth_image_3_dilation{MemoryType::kUnified};
  depth_image_3_dilation.copyFromAsync(depth_image, *cuda_stream_);
  depth_preprocessor_ptr_->dilateInvalidRegionsAsync(3,
                                                     &depth_image_3_dilation);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
  EXPECT_EQ(getImageSumOnCPU(depth_image_3_dilation),
            get_expected_image_sum(3));

  // 4 dilation
  DepthImage depth_image_4_dilation{MemoryType::kUnified};
  depth_image_4_dilation.copyFromAsync(depth_image, *cuda_stream_);
  depth_preprocessor_ptr_->dilateInvalidRegionsAsync(4,
                                                     &depth_image_4_dilation);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());
  EXPECT_EQ(getImageSumOnCPU(depth_image_4_dilation),
            get_expected_image_sum(4));

  // Check the result
  if (writeDebugOutput) {
    std::cout << "depth_image: " << std::endl;
    printImageToConsole(depth_image);
    std::cout << std::endl << "depth_image_1_dilation: " << std::endl;
    printImageToConsole(depth_image_1_dilation);
    std::cout << std::endl << "depth_image_2_dilation: " << std::endl;
    printImageToConsole(depth_image_2_dilation);
    std::cout << std::endl << "depth_image_3_dilation: " << std::endl;
    printImageToConsole(depth_image_3_dilation);
    std::cout << std::endl << "depth_image_4_dilation: " << std::endl;
    printImageToConsole(depth_image_4_dilation);
  }
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
