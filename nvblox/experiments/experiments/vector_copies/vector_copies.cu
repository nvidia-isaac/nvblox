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
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "nvblox/core/unified_vector.h"
#include "nvblox/tests/increment_on_gpu.h"
#include "nvblox/utils/timing.h"

#include "vector_copies.h"

using namespace nvblox;

void runVectorCopyExperiments(int num_runs) {
  constexpr size_t kVectorSize = 1000000;

  for (int i = 0; i < num_runs; i++) {
    // Experiment 1a: copy vector host-to-device with thrust.
    timing::Timer exp_initialize_timer("exp/initialize");
    std::vector<int> start_vec(kVectorSize, 5);
    exp_initialize_timer.Stop();

    {
      timing::Timer exp1a_timer("exp1a/thrust/host_to_device");
      thrust::device_vector<int> vec_1a(kVectorSize);
      thrust::copy(start_vec.begin(), start_vec.end(), vec_1a.begin());
      exp1a_timer.Stop();

      timing::Timer exp1a_kernel_timer("exp1a/kernel");
      test_utils::incrementOnGPU(kVectorSize,
                                 thrust::raw_pointer_cast(vec_1a.data()));
      exp1a_kernel_timer.Stop();

      // Experiment 1b: copy vector device-to-device with thrust.
      timing::Timer exp1b_timer("exp1b/thrust/device_to_device");
      thrust::device_vector<int> vec_1b(kVectorSize);
      thrust::copy(vec_1a.begin(), vec_1a.end(), vec_1b.begin());
      exp1b_timer.Stop();

      timing::Timer exp1b_kernel_timer("exp1b/kernel");
      test_utils::incrementOnGPU(kVectorSize,
                                 thrust::raw_pointer_cast(vec_1b.data()));
      exp1b_kernel_timer.Stop();

      // Experiment 1c: copy vector device-to-host with thrust.
      timing::Timer exp1c_timer("exp1c/thrust/device_to_host");
      std::vector<int> vec_1c(kVectorSize);
      thrust::copy(vec_1a.begin(), vec_1a.end(), vec_1c.begin());
      exp1c_timer.Stop();
    }

    {
      // Experiment 2a: copy vector host-to-device with cuda memcpy.
      timing::Timer exp2a_timer("exp2a/cuda/host_to_device");
      thrust::device_vector<int> vec_2a(kVectorSize);
      cudaMemcpy(thrust::raw_pointer_cast(vec_2a.data()), start_vec.data(),
                 kVectorSize * sizeof(int), cudaMemcpyHostToDevice);
      exp2a_timer.Stop();

      timing::Timer exp2a_kernel_timer("exp2a/kernel");
      test_utils::incrementOnGPU(kVectorSize,
                                 thrust::raw_pointer_cast(vec_2a.data()));
      exp2a_kernel_timer.Stop();

      // Experiment 2b: copy vector device-to-device with cuda memcpy.
      timing::Timer exp2b_timer("exp2b/cuda/device_to_device");
      thrust::device_vector<int> vec_2b(kVectorSize);
      cudaMemcpy(thrust::raw_pointer_cast(vec_2b.data()),
                 thrust::raw_pointer_cast(vec_2a.data()),
                 kVectorSize * sizeof(int), cudaMemcpyDeviceToDevice);
      exp2b_timer.Stop();

      timing::Timer exp2b_kernel_timer("exp2b/kernel");
      test_utils::incrementOnGPU(kVectorSize,
                                 thrust::raw_pointer_cast(vec_2b.data()));
      exp2b_kernel_timer.Stop();

      // Experiment 2c: copy vector device-to-host with cuda memcpy.
      timing::Timer exp2c_timer("exp2c/cuda/device_to_host");
      std::vector<int> vec_2c(kVectorSize);
      cudaMemcpy(vec_2c.data(), thrust::raw_pointer_cast(vec_2b.data()),
                 kVectorSize * sizeof(int), cudaMemcpyDeviceToHost);
      exp2c_timer.Stop();
    }

    {
      // Experiment 3: create a unified vector and use it in a kernel.
      timing::Timer exp3_initialize_timer("exp3/unified/initialize");
      unified_vector<int> unified_vec(kVectorSize, 5);
      exp3_initialize_timer.Stop();

      timing::Timer exp3_kernel_timer("exp3/unified/kernel");
      test_utils::incrementOnGPU(kVectorSize, unified_vec.data());
      exp3_kernel_timer.Stop();

      timing::Timer exp3_increment_timer("exp3/unified/cpu_increment");
      for (size_t i = 0; i < kVectorSize; i++) {
        unified_vec[i]++;
      }
      exp3_increment_timer.Stop();

      timing::Timer exp3_host_increment_timer("exp3/host/cpu_increment");
      for (size_t i = 0; i < kVectorSize; i++) {
        start_vec[i]++;
      }
      exp3_host_increment_timer.Stop();
    }
    // Experiment 4: is it faster to create an array on host, copy its value to
    // GPU, delete it or use unified memory?
    // 4a: baseline: device copy.
    {
      timing::Timer exp4a("exp4a/device_mem");
      std::vector<int> exp4a_vec(kVectorSize, 1);
      int* exp4a_dev;
      cudaMalloc(&exp4a_dev, kVectorSize * sizeof(int));
      cudaMemcpy(exp4a_dev, exp4a_vec.data(), kVectorSize * sizeof(int),
                 cudaMemcpyHostToDevice);
      test_utils::incrementOnGPU(kVectorSize, exp4a_dev);
      cudaFree(exp4a_dev);
      exp4a.Stop();
    }

    // 4b: unified without anything special, using memset to initialize.
    {
      timing::Timer exp4b("exp4b/unified_mem");
      unified_vector<int> unified_vec(kVectorSize);
      memset(unified_vec.data(), kVectorSize * sizeof(int), 1);
      test_utils::incrementOnGPU(kVectorSize, unified_vec.data());
      exp4b.Stop();
    }

    // 4c: unified vector using the unified vector code (on CPU) to initialize.
    {
      timing::Timer exp4c("exp4c/unified_cpu_set");
      unified_vector<int> unified_vec(kVectorSize, 1);
      // unified_vec.toGPU();
      test_utils::incrementOnGPU(kVectorSize, unified_vec.data());
      exp4c.Stop();
    }
  }
}