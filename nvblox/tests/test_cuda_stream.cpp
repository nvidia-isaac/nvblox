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
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/tests/increment_on_gpu.h"

using namespace nvblox;

void incrementOnStreamInFunction(CudaStream* stream_ptr, int* int_dev_ptr) {
  test_utils::incrementOnStream(int_dev_ptr, stream_ptr);
}

TEST(CudaStreamTest, OwningStreamTest) {
  CudaStreamOwning cuda_stream;

  auto int_ptr = make_unified<int>(MemoryType::kUnified, 0);
  EXPECT_EQ(*int_ptr, 0);
  test_utils::incrementOnStream(int_ptr.get(), &cuda_stream);
  EXPECT_EQ(*int_ptr, 1);

  // Test the we can use a stream through the base class interface.
  incrementOnStreamInFunction(&cuda_stream, int_ptr.get());
  EXPECT_EQ(*int_ptr, 2);

  // NOTE(alex.millane): We would like to test that the stream is properly
  // destoyed when the owning allocator is, but this is currently impossible
  // with the CUDA runtime API.
}

TEST(CudaStreamTest, NonOwningStreamTest) {
  // Setting up a raw CUDA stream
  cudaStream_t raw_cuda_stream;
  checkCudaErrors(cudaStreamCreate(&raw_cuda_stream));

  // Wrapping it in a non-owning allocator
  CudaStreamNonOwning cuda_stream(&raw_cuda_stream);

  auto int_ptr = make_unified<int>(MemoryType::kUnified, 0);
  EXPECT_EQ(*int_ptr, 0);
  test_utils::incrementOnStream(int_ptr.get(), &cuda_stream);
  EXPECT_EQ(*int_ptr, 1);

  // Test the we can use a stream through the base class interface.
  incrementOnStreamInFunction(&cuda_stream, int_ptr.get());
  EXPECT_EQ(*int_ptr, 2);

  // Test that we can operate on a copy
  auto cuda_stream_copy = cuda_stream;
  test_utils::incrementOnStream(int_ptr.get(), &cuda_stream);
  EXPECT_EQ(*int_ptr, 3);

  // Tear down the raw stream
  checkCudaErrors(cudaStreamDestroy(raw_cuda_stream));
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
