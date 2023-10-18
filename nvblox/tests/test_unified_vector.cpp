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

#include <vector>

#include <cuda_runtime.h>

#include "nvblox/core/internal/warmup_cuda.h"
#include "nvblox/core/unified_vector.h"

#include "nvblox/tests/increment_on_gpu.h"
#include "nvblox/tests/test_utils_cuda.h"

using namespace nvblox;

cudaMemoryType getPointerMemoryType(void* pointer) {
  // Check that the memory is allocated.
  cudaPointerAttributes attributes;
  cudaError_t error = cudaPointerGetAttributes(&attributes, pointer);
  EXPECT_EQ(error, cudaSuccess);
  return attributes.type;
}

TEST(UnifiedVectorTest, EmptyTest) {
  unified_vector<float> vec;
  EXPECT_TRUE(vec.empty());
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_EQ(vec.size(), 0);
}

TEST(UnifiedVectorTest, ClearTest) {
  unified_vector<float> vec(100, 10);
  EXPECT_FALSE(vec.empty());
  EXPECT_NE(vec.data(), nullptr);
  EXPECT_EQ(vec.size(), 100);
  EXPECT_EQ(vec[0], 10);
  EXPECT_EQ(vec[99], 10);

  vec.clear();
  EXPECT_TRUE(vec.empty());
  EXPECT_EQ(vec.data(), nullptr);
  EXPECT_EQ(vec.size(), 0);
}

TEST(UnifiedVectorTest, ClearNoDeallocTest) {
  unified_vector<float> vec(100);

  const float* ptr = vec.data();
  EXPECT_NE(ptr, nullptr);

  vec.clearNoDealloc();

  EXPECT_EQ(vec.data(), ptr);
  EXPECT_TRUE(vec.empty());
  EXPECT_EQ(vec.size(), 0);
  EXPECT_EQ(vec.capacity(), 100);
}

TEST(UnifiedVectorTest, PushBackTest) {
  unified_vector<size_t> vec;
  constexpr size_t kMaxSize = 999999;
  for (size_t i = 0; i < kMaxSize; i++) {
    vec.push_back(i);
  }
  EXPECT_FALSE(vec.empty());
  EXPECT_NE(vec.data(), nullptr);
  EXPECT_EQ(vec.size(), kMaxSize);

  for (size_t i = 0; i < kMaxSize; i++) {
    EXPECT_EQ(vec[i], i);
  }
  EXPECT_GE(vec.capacity(), vec.size());
}

TEST(UnifiedVectorTest, AssignmentTest) {
  unified_vector<size_t> vec;
  constexpr size_t kMaxSize = 10;
  for (size_t i = 0; i < kMaxSize; i++) {
    vec.push_back(i);
  }

  // Copy the vector over;
  unified_vector<size_t> vec2;
  vec2.copyFrom(vec);

  EXPECT_FALSE(vec2.empty());
  EXPECT_NE(vec2.data(), nullptr);
  EXPECT_EQ(vec2.size(), kMaxSize);

  for (size_t i = 0; i < kMaxSize; i++) {
    EXPECT_EQ(vec2[i], i);
  }
  EXPECT_GE(vec2.capacity(), vec.size());
}

TEST(UnifiedVectorTest, IteratorTest) {
  unified_vector<size_t> vec;
  constexpr size_t kMaxSize = 10;
  for (size_t i = 0; i < kMaxSize; i++) {
    vec.push_back(i);
  }

  size_t i = 0;
  for (size_t num : vec) {
    EXPECT_EQ(num, i++);
  }
  EXPECT_EQ(i, kMaxSize);
}

TEST(UnifiedVectorTest, CpuGpuReadWrite) {
  const float value = 12.3f;

  const size_t kVectorSize = 100;
  unified_vector<float> vec;
  vec.resize(kVectorSize);
  test_utils::fillVectorWithConstant(value, &vec);

  // Use the operator to check on the HOST
  for (size_t i = 0; i < kVectorSize; i++) {
    EXPECT_EQ(vec[i], value);
  }

  // Check on the DEVICE
  EXPECT_TRUE(test_utils::checkVectorAllConstant(vec, value));
}

TEST(UnifiedVectorTest, SetZeroTest) {
  constexpr size_t kSize = 100;
  unified_vector<int> vec(kSize, 1, MemoryType::kHost);
  for (size_t i = 0; i < kSize; i++) {
    EXPECT_EQ(vec[i], 1);
  }
  vec.setZero();
  for (size_t i = 0; i < kSize; i++) {
    EXPECT_EQ(vec[i], 0);
  }
}

TEST(UnifiedVectorTest, HostToDeviceToHostCopy) {
  constexpr size_t kSize = 100;
  unified_vector<int> vec(kSize, MemoryType::kHost);
  EXPECT_EQ(getPointerMemoryType(vec.data()),
            cudaMemoryType::cudaMemoryTypeHost);
  for (size_t i = 0; i < kSize; i++) {
    vec[i] = i;
  }

  // Copy over to unified memory.
  unified_vector<int> vec_unified(MemoryType::kUnified);
  vec_unified.copyFrom(vec);
  EXPECT_EQ(getPointerMemoryType(vec_unified.data()),
            cudaMemoryType::cudaMemoryTypeManaged);
  for (size_t i = 0; i < kSize; i++) {
    EXPECT_EQ(vec_unified[i], i);
    // Modify it for later.
    vec_unified[i] = i - 1;
  }

  // Copy over to device memory.
  unified_vector<int> vec_device(MemoryType::kDevice);
  vec_device.copyFrom(vec_unified);
  EXPECT_EQ(getPointerMemoryType(vec_device.data()),
            cudaMemoryType::cudaMemoryTypeDevice);

  unified_vector<int> vec_host(MemoryType::kHost);
  vec_host.copyFrom(vec_device);
  EXPECT_EQ(getPointerMemoryType(vec_host.data()),
            cudaMemoryType::cudaMemoryTypeHost);
  for (size_t i = 0; i < kSize; i++) {
    EXPECT_EQ(vec_host[i], i - 1);
  }
}

template <typename VectorType>
void checkAllConstantCPU(const VectorType& vec, const int value) {
  for (auto i : vec) {
    EXPECT_EQ(i, value);
  }
}

TEST(UnifiedVectorTest, HostAndDeviceVectors) {
  // Host vector
  constexpr int kNumElems = 100;
  host_vector<int> vec_host_1(kNumElems, 1);
  checkAllConstantCPU(vec_host_1, 1);

  // To device
  device_vector<int> vec_device;
  vec_device.copyFrom(vec_host_1);
  EXPECT_TRUE(test_utils::checkAllConstant(vec_device.data(), 1, kNumElems));
  EXPECT_FALSE(test_utils::checkAllConstant(vec_device.data(), 2, kNumElems));
  test_utils::incrementOnGPU(kNumElems, vec_device.data());
  EXPECT_TRUE(test_utils::checkAllConstant(vec_device.data(), 2, kNumElems));

  // Back to host
  host_vector<int> vec_host_2;
  vec_host_2.copyFrom(vec_device);
  checkAllConstantCPU(vec_host_2, 2);

  // Conversion to std::vector
  std::vector<int> std_vec_host_1 = vec_device.toVector();
  checkAllConstantCPU(std_vec_host_1, 2);

  // Conversion from std::vector
  std::vector<int> std_vec_host_2(kNumElems, 3);
  checkAllConstantCPU(std_vec_host_2, 3);
  device_vector<int> vec_device_2;
  vec_device_2.copyFrom(std_vec_host_2);
  EXPECT_TRUE(test_utils::checkAllConstant(vec_device_2.data(), 3, kNumElems));

  // Checking conversion from base class.
  host_vector<int> std_vec_host_3(kNumElems, 4);
  const unified_vector<int>& base_class_reference = std_vec_host_3;
  checkAllConstantCPU(base_class_reference, 4);
  device_vector<int> vec_device_3;
  vec_device_3.copyFrom(base_class_reference);
  EXPECT_TRUE(test_utils::checkAllConstant(vec_device_3.data(), 4, kNumElems));
}

TEST(UnifiedVectorTest, BoolTest) {
  // Setup test vector (true and false in alternating fashion)
  unified_vector<bool> vec_unified(100);
  for (size_t i = 0; i < vec_unified.size(); i++) {
    if (i % 2 == 0) {
      vec_unified[i] = true;
    } else {
      vec_unified[i] = false;
    }
  }
  // Convert
  std::vector<bool> vec_std = vec_unified.toVector();
  // Check
  for (int i = 0; i < 100; i++) {
    EXPECT_EQ(vec_unified[i], vec_std[i]);
  }
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
