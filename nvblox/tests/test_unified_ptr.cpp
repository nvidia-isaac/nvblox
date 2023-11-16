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

#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"
#include "nvblox/map/accessors.h"
#include "nvblox/map/common_names.h"

#include "nvblox/tests/increment_on_gpu.h"
#include "nvblox/tests/test_utils_cuda.h"

using namespace nvblox;

void setToConstantOnCPU(const int value, const int num_elements, int ints[]) {
  for (int i = 0; i < num_elements; i++) {
    ints[i] = value;
  }
}

void incrementOnCPU(const int num_elements, int ints[]) {
  for (int i = 0; i < num_elements; i++) {
    ints[i]++;
  }
}

template <typename T>
void expect_cuda_freed(T* ptr) {
  // Note(alexmillane): Whether or not this call returns an error seems to vary
  // between machines... Hopefully the checks below indicate that the memory has
  // been freed.
  cudaPointerAttributes attributes;
  checkCudaErrors(cudaPointerGetAttributes(&attributes, ptr));
  EXPECT_EQ(attributes.type, cudaMemoryType::cudaMemoryTypeUnregistered);
  EXPECT_EQ(attributes.devicePointer, nullptr);
  EXPECT_EQ(attributes.hostPointer, nullptr);
}

TEST(UnifiedPointerTest, IntTest) {
  unified_ptr<int> hello = make_unified<int>(3);

  EXPECT_TRUE(hello);
  EXPECT_EQ(*hello, 3);
  (*hello)++;
  EXPECT_EQ(*hello, 4);

  test_utils::incrementOnGPU(hello.get());

  EXPECT_EQ(*hello, 5);

  hello.reset();
  EXPECT_FALSE(hello);
  EXPECT_EQ(hello.get(), nullptr);
}

TEST(UnifiedPointerTest, ObjectTest) {
  unified_ptr<std::array<int, 100>> array_of_ints =
      make_unified<std::array<int, 100>>();

  EXPECT_TRUE(array_of_ints);
  (*array_of_ints)[3] = 4;
  EXPECT_EQ((*array_of_ints)[3], 4);
}

TEST(UnifiedPointerTest, EmptyTest) {
  unified_ptr<Eigen::Vector3f> vec;
  EXPECT_FALSE(vec);
}

TEST(UnifiedPointerTest, MemoryTest) {
  int* raw_ptr;
  {
    // Put this in a disappearing scope.
    unified_ptr<int> dummy_ptr(make_unified<int>(100));
    raw_ptr = dummy_ptr.get();

    // Check that the memory is allocated.
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, raw_ptr);
    EXPECT_EQ(error, cudaSuccess);
    EXPECT_EQ(attributes.type, cudaMemoryType::cudaMemoryTypeManaged);
  }

  // Make sure the memory no longer exists now it's out of scope.
  cudaPointerAttributes attributes;
  checkCudaErrors(cudaPointerGetAttributes(&attributes, raw_ptr));
  // Note(alexmillane): Whether or not this call returns an error seems to vary
  // between machines... Hopefully the checks below indicate that the memory has
  // been freed. EXPECT_EQ(error, cudaErrorInvalidValue);
  EXPECT_EQ(attributes.type, cudaMemoryType::cudaMemoryTypeUnregistered);
  EXPECT_EQ(attributes.devicePointer, nullptr);
  EXPECT_EQ(attributes.hostPointer, nullptr);
}

TEST(UnifiedPointerTest, ArrayTest) {
  constexpr int kNumElements = 1000;
  unified_ptr<int[]> goodbye = make_unified<int[]>(kNumElements);

  constexpr int kValue = 100;
  setToConstantOnCPU(kValue, kNumElements, goodbye.get());

  for (int i = 0; i < kNumElements; i++) {
    EXPECT_EQ(goodbye[i], kValue);
  }

  incrementOnCPU(kNumElements, goodbye.get());

  for (int i = 0; i < kNumElements; i++) {
    EXPECT_EQ(goodbye[i], kValue + 1);
  }

  test_utils::incrementOnGPU(kNumElements, goodbye.get());

  for (int i = 0; i < kNumElements; i++) {
    EXPECT_EQ(goodbye[i], kValue + 2);
  }

  int* raw_ptr = goodbye.get();
  goodbye.reset();
  EXPECT_FALSE(goodbye);
  EXPECT_EQ(goodbye.get(), nullptr);
  // Make sure the memory no longer exists now it's out of scope.
  expect_cuda_freed(raw_ptr);
}

TEST(UnifiedPointerTest, HostTest) {
  auto int_host_ptr = make_unified<int>(MemoryType::kHost, 1);
  EXPECT_EQ(*int_host_ptr, 1);
  EXPECT_NE(*int_host_ptr, 2);

  cudaPointerAttributes attributes;
  checkCudaErrors(cudaPointerGetAttributes(&attributes, int_host_ptr.get()));
  EXPECT_EQ(attributes.type, cudaMemoryType::cudaMemoryTypeHost);

  int* raw_ptr = int_host_ptr.get();
  int_host_ptr.reset();
  EXPECT_FALSE(int_host_ptr);
  EXPECT_EQ(int_host_ptr.get(), nullptr);
  expect_cuda_freed(raw_ptr);
}

TEST(UnifiedPointerTest, CloneTest) {
  // Host-only cloning
  auto int_ptr_1 = make_unified<int>(MemoryType::kHost, 1);
  EXPECT_EQ(*int_ptr_1, 1);
  auto int_ptr_2 = int_ptr_1.clone();
  EXPECT_EQ(*int_ptr_2, 1);
  *int_ptr_2 += 1;
  EXPECT_EQ(*int_ptr_2, 2);
  EXPECT_EQ(*int_ptr_1, 1);

  // Host-to-device-to-unified cloning
  auto int_ptr_3 = int_ptr_1.clone(MemoryType::kDevice);
  test_utils::incrementOnGPU(int_ptr_3.get());
  auto int_ptr_4 = int_ptr_3.clone(MemoryType::kUnified);
  EXPECT_EQ(*int_ptr_4, 2);

  // Array cloning
  constexpr int kNumElems = 10;
  auto int_ptr_5 = make_unified<int[]>(kNumElems, MemoryType::kDevice);
  test_utils::fillWithConstant(1, kNumElems, int_ptr_5.get());
  EXPECT_TRUE(test_utils::checkAllConstant(int_ptr_5.get(), 1, kNumElems));
  test_utils::incrementOnGPU(kNumElems, int_ptr_5.get());
  EXPECT_TRUE(test_utils::checkAllConstant(int_ptr_5.get(), 2, kNumElems));
  test_utils::incrementOnGPU(kNumElems, int_ptr_5.get());
  auto int_ptr_6 = int_ptr_5.clone(MemoryType::kHost);

  for (int i = 0; i < kNumElems; i++) {
    EXPECT_EQ(int_ptr_6[i], 3);
  }
}

TEST(UnifiedPointerTest, CopyFrom) {
  constexpr int32_t kTestValue = 123456;
  auto ptr_host = make_unified<int>(MemoryType::kHost, kTestValue);
  auto ptr_device = make_unified<int>(MemoryType::kDevice);

  ptr_device.copyFrom(ptr_host);
  auto ptr_host2 = make_unified<int>(MemoryType::kHost);
  ptr_host2.copyFrom(ptr_device);

  EXPECT_EQ(*ptr_host2, kTestValue);
}

TEST(UnifiedPointerTest, CopyTo) {
  constexpr int32_t kTestValue = 123456;
  auto ptr_host = make_unified<int>(MemoryType::kHost, kTestValue);
  auto ptr_device = make_unified<int>(MemoryType::kDevice);

  ptr_host.copyTo(ptr_device);
  auto ptr_host2 = make_unified<int>(MemoryType::kHost);
  ptr_device.copyTo(ptr_host2);

  EXPECT_EQ(*ptr_host2, kTestValue);
}

TEST(UnifiedPointerTest, CloneConstBlock) {
  constexpr float kBlockSize = 1.0f;
  TsdfLayer layer(kBlockSize, MemoryType::kDevice);
  auto block_ptr = layer.allocateBlockAtIndex({0, 0, 0});

  const TsdfLayer& layer_const_ref = layer;

  TsdfBlock::Ptr cloned_block =
      layer_const_ref.getBlockAtIndex({0, 0, 0}).clone(MemoryType::kHost);

  auto expect_voxel_zero_lambda = [](const Index3D&,
                                     const TsdfVoxel* voxel) -> void {
    EXPECT_EQ(voxel->distance, 0.0f);
    EXPECT_EQ(voxel->weight, 0.0f);
  };
  callFunctionOnAllVoxels<TsdfVoxel>(*cloned_block, expect_voxel_zero_lambda);
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
