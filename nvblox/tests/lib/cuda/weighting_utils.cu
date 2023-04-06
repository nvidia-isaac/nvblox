#include "nvblox/integrators/weighting_function.h"
#include "nvblox/tests/weighting_utils.h"

namespace nvblox {
namespace test_utils {

__global__ void computeWeightKernel(WeightingFunction* function,
                                    float point_distance_from_camera,
                                    float voxel_distance_from_camera,
                                    float truncation_distance, float* output) {
  *output = (*function)(point_distance_from_camera, voxel_distance_from_camera,
                        truncation_distance);
}

float computeWeight(unified_ptr<WeightingFunction>& weighting_function,
                    float point_distance_from_camera,
                    float voxel_distance_from_camera,
                    float truncation_distance) {
  unified_ptr<float> output = make_unified<float>(MemoryType::kDevice);

  computeWeightKernel<<<1, 1>>>(
      weighting_function.get(), point_distance_from_camera,
      voxel_distance_from_camera, truncation_distance, output.get());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaPeekAtLastError());

  unified_ptr<float> output_host = make_unified<float>(MemoryType::kHost);
  output.copyTo(output_host);

  return *output_host;
}

}  // namespace test_utils
}  // namespace nvblox
