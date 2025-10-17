/*
Copyright 2024 NVIDIA CORPORATION

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

#include <nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh>

#include "nvblox/conversions/esdf_slice_conversions.h"

namespace nvblox {
namespace conversions {

constexpr int8_t kOccupancyGridUnknownValue = -1;

// Calling rules:
// - Should be called with 2D grid of thread-blocks where the total number of
// threads in each dimension exceeds the number of pixels in each image
// dimension ie:
// - blockIdx.x * blockDim.x + threadIdx.x > cols
// - blockIdx.y * blockDim.y + threadIdx.y > rows
// - We assume that theres enough space in the pointcloud to store a point per
// pixel if required.
__global__ void occupancyGridFromSliceImageKernel(
    const float* slice_image_ptr,         // NOLINT
    const int rows,                       // NOLINT
    const int cols,                       // NOLINT
    const float unknown_value,            // NOLINT
    int8_t* occupancy_grid_device_ptr) {  // NOLINT
  // Get the pixel addressed by this thread.
  const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int row_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (col_idx >= cols || row_idx >= rows) {
    return;
  }

  // Access the slice
  const float pixel_value =
      image::access(row_idx, col_idx, cols, slice_image_ptr);

  constexpr float kEps = 1e-2;
  constexpr int8_t kOccupiedValue = 100;

  // If the distance is under epsilon (near zero), we're in occupied space so
  // set it to kOccupiedValue (via implicit cast from Bool(True) to int(1)).
  int8_t value = (pixel_value < kEps) * kOccupiedValue;
  // If the value is approximately equal to the constant signifying unknown,
  // set it to unknown in ros (-1).
  if (fabsf(pixel_value - unknown_value) < kEps) {
    value = kOccupancyGridUnknownValue;
  }

  // Write the point to the ouput
  image::access(row_idx, col_idx, cols, occupancy_grid_device_ptr) = value;
}

EsdfSliceConverter::EsdfSliceConverter()
    : EsdfSliceConverter(std::make_shared<CudaStreamOwning>()) {}

EsdfSliceConverter::EsdfSliceConverter(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream), esdf_slicer_(cuda_stream) {}

void EsdfSliceConverter::sliceLayerToDistanceImage(
    const EsdfLayer& layer, float slice_height, float unknown_value,
    Image<float>* output_image, AxisAlignedBoundingBox* aabb) {
  CHECK_NOTNULL(aabb);
  *aabb = esdf_slicer_.getAabbOfLayerAtHeight(layer, slice_height);
  esdf_slicer_.sliceLayerToDistanceImage(layer, slice_height, unknown_value,
                                         *aabb, output_image);
}

void EsdfSliceConverter::occupancyGridFromSliceImage(
    const Image<float>& slice_image, signed char* occupancy_grid_data,
    const float unknown_value) {
  CHECK_NOTNULL(occupancy_grid_data);

  // Can happen that this function is called before the ESDF contains data.
  // Return without doing anything if that happens.
  if (slice_image.numel() <= 0 || slice_image.rows() <= 0 ||
      slice_image.cols() <= 0) {
    return;
  }

  const int width = slice_image.cols();
  const int height = slice_image.rows();

  // Allocate device-side scratch pad
  occupancy_grid_device_.reserveAsync(width * height, *cuda_stream_);

  // Call CUDA kernel to convert from float distance to int8 occupancy.
  // Kernel
  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(
      slice_image.cols() / kThreadsPerThreadBlock.x + 1,  // NOLINT
      slice_image.rows() / kThreadsPerThreadBlock.y + 1,  // NOLINT
      1);
  occupancyGridFromSliceImageKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                                      *cuda_stream_>>>(
      slice_image.dataConstPtr(),    // NOLINT
      slice_image.rows(),            // NOLINT
      slice_image.cols(),            // NOLINT
      unknown_value,                 // NOLINT
      occupancy_grid_device_.data()  // NOLINT
  );
  checkCudaErrors(cudaPeekAtLastError());

  // Copy into the message
  checkCudaErrors(cudaMemcpyAsync(
      occupancy_grid_data, occupancy_grid_device_.data(),
      width * height * sizeof(int8_t), cudaMemcpyDefault, *cuda_stream_));
  cuda_stream_->synchronize();
}

}  // namespace conversions
}  // namespace nvblox
