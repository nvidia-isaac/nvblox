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
#include "nvblox/integrators/view_calculator.h"

#include "nvblox/core/hash.h"
#include "nvblox/core/indexing.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/rays/ray_caster.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

/// NOTE(gogojjh): define template function
template std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycast(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    const float block_size, const float truncation_distance_m,
    const float max_integration_distance_m);

template std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycast(
    const DepthImage& depth_frame, const Transform& T_L_C,
    const CameraPinhole& camera, const float block_size,
    const float truncation_distance_m, const float max_integration_distance_m);

///////////////////////////////////////////////////////////////
ViewCalculator::ViewCalculator() { cudaStreamCreate(&cuda_stream_); }
ViewCalculator::~ViewCalculator() { cudaStreamDestroy(cuda_stream_); }

unsigned int ViewCalculator::raycast_subsampling_factor() const {
  return raycast_subsampling_factor_;
}

void ViewCalculator::raycast_subsampling_factor(
    unsigned int raycast_subsampling_factor) {
  CHECK_GT(raycast_subsampling_factor, 0);
  raycast_subsampling_factor_ = raycast_subsampling_factor;
}

// AABB linear indexing
// - We index in x-major, i.e. x is varied first, then y, then z.
// - Linear indexing within an AABB is relative and starts at zero. This is
//   not true for AABB 3D indexing which is w.r.t. the layer origin.
__host__ __device__ inline size_t layerIndexToAabbLinearIndex(
    const Index3D& index, const Index3D& aabb_min, const Index3D& aabb_size) {
  const Index3D index_shifted = index - aabb_min;
  return index_shifted.x() +                                 // NOLINT
         index_shifted.y() * aabb_size.x() +                 // NOLINT
         index_shifted.z() * aabb_size.x() * aabb_size.y();  // NOLINT
}

__host__ __device__ inline Index3D aabbLinearIndexToLayerIndex(
    const size_t lin_idx, const Index3D& aabb_min, const Index3D& aabb_size) {
  const Index3D index(lin_idx % aabb_size.x(),                     // NOLINT
                      (lin_idx / aabb_size.x()) % aabb_size.y(),   // NOLINT
                      lin_idx / (aabb_size.x() * aabb_size.y()));  // NOLINT
  return index + aabb_min;
}

__device__ void setIndexUpdated(const Index3D& index_to_update,
                                const Index3D& aabb_min,
                                const Index3D& aabb_size, bool* aabb_updated) {
  const size_t linear_size = aabb_size.x() * aabb_size.y() * aabb_size.z();
  const size_t lin_idx =
      layerIndexToAabbLinearIndex(index_to_update, aabb_min, aabb_size);
  if (lin_idx < linear_size) {
    aabb_updated[lin_idx] = true;
  }
}

template <typename T>
void convertAabbUpdatedToVector(const Index3D& aabb_min,
                                const Index3D& aabb_size,
                                size_t aabb_linear_size, bool* aabb_updated,
                                T* indices) {
  indices->reserve(aabb_linear_size);
  for (size_t i = 0; i < aabb_linear_size; i++) {
    if (aabb_updated[i]) {
      indices->push_back(aabbLinearIndexToLayerIndex(i, aabb_min, aabb_size));
    }
  }
}

template <typename SensorType>
__global__ void getBlockIndicesInImageKernel(
    const Transform T_L_C, const SensorType camera, const float* image,
    int rows, int cols, const float block_size,
    const float max_integration_distance_m, const float truncation_distance_m,
    const Index3D aabb_min, const Index3D aabb_size, bool* aabb_updated) {
  // First, figure out which pixel we're in.
  int pixel_row = blockIdx.x * blockDim.x + threadIdx.x;
  int pixel_col = blockIdx.y * blockDim.y + threadIdx.y;

  // Hooray we do nothing.
  if (pixel_row >= rows || pixel_col >= cols) {
    return;
  }

  // Look up the pixel we care about.
  float depth = image::access<float>(pixel_row, pixel_col, cols, image);
  if (depth <= 0.0f) {
    return;
  }
  if (max_integration_distance_m > 0.0f && depth > max_integration_distance_m) {
    depth = max_integration_distance_m;
  }

  // Ok now project this thing into space.
  Vector3f p_C = (depth + truncation_distance_m) *
                 camera.vectorFromPixelIndices(Index2D(pixel_col, pixel_row));
  Vector3f p_L = T_L_C * p_C;

  // Now we have the position of the thing in space. Now we need the block
  // index.
  Index3D block_index = getBlockIndexFromPositionInLayer(block_size, p_L);
  setIndexUpdated(block_index, aabb_min, aabb_size, aabb_updated);
}

__global__ void raycastToBlocksKernel(int num_blocks, Index3D* block_indices,
                                      const Transform T_L_C, float block_size,
                                      const Index3D aabb_min,
                                      const Index3D aabb_size,
                                      bool* aabb_updated) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int corner_index = threadIdx.y;

  if (index >= num_blocks) {
    return;
  }

  constexpr float corner_increment_table[9][3] = {
      {0.0f, 0.0f, 0.0f},  // NOLINT
      {1.0f, 0.0f, 0.0f},  // NOLINT
      {0.0f, 1.0f, 0.0f},  // NOLINT
      {0.0f, 0.0f, 1.0f},  // NOLINT
      {1.0f, 1.0f, 0.0f},  // NOLINT
      {1.0f, 0.0f, 1.0f},  // NOLINT
      {0.0f, 1.0f, 1.0f},  // NOLINT
      {1.0f, 1.0f, 1.0f},  // NOLINT
      {0.5f, 0.5f, 0.5f},  // NOLINT
  };

  const Vector3f increment(corner_increment_table[corner_index][0],
                           corner_increment_table[corner_index][1],
                           corner_increment_table[corner_index][2]);

  const Index3D& block_index = block_indices[index];

  RayCaster raycaster(T_L_C.translation() / block_size,
                      block_index.cast<float>() + increment);
  Index3D ray_index;
  while (raycaster.nextRayIndex(&ray_index)) {
    setIndexUpdated(ray_index, aabb_min, aabb_size, aabb_updated);
  }
}

template <typename SensorType>
__global__ void combinedBlockIndicesInImageKernel(
    const Transform T_L_C, const SensorType camera, const float* image,
    int rows, int cols, const float block_size,
    const float max_integration_distance_m, const float truncation_distance_m,
    int raycast_subsampling_factor, const Index3D aabb_min,
    const Index3D aabb_size, bool* aabb_updated) {
  // First, figure out which pixel we're in.
  const int ray_idx_row = blockIdx.x * blockDim.x + threadIdx.x;
  const int ray_idx_col = blockIdx.y * blockDim.y + threadIdx.y;
  int pixel_row = ray_idx_row * raycast_subsampling_factor;
  int pixel_col = ray_idx_col * raycast_subsampling_factor;

  // Hooray we do nothing.
  if (pixel_row >= (rows + raycast_subsampling_factor - 1) ||
      pixel_col >= (cols + raycast_subsampling_factor - 1)) {
    return;
  } else {
    // Move remaining overhanging pixels back to the borders.
    if (pixel_row >= rows) {
      pixel_row = rows - 1;
    }
    if (pixel_col >= cols) {
      pixel_col = cols - 1;
    }
  }

  // Look up the pixel we care about.
  float depth = image::access<float>(pixel_row, pixel_col, cols, image);
  if (depth <= 0.0f) {
    return;
  }
  if (max_integration_distance_m > 0.0f && depth > max_integration_distance_m) {
    depth = max_integration_distance_m;
  }

  // Ok now project this thing into space.
  // in the camera coordinate
  Vector3f p_C = (depth + truncation_distance_m) *
                 camera.vectorFromPixelIndices(Index2D(pixel_col, pixel_row));
  Vector3f p_L = T_L_C * p_C;

  // Now we have the position of the thing in space. Now we need the block
  // index.
  Index3D block_index = getBlockIndexFromPositionInLayer(block_size, p_L);
  setIndexUpdated(block_index, aabb_min, aabb_size, aabb_updated);

  // Ok raycast to the correct point in the block.
  RayCaster raycaster(T_L_C.translation() / block_size, p_L / block_size);
  Index3D ray_index = Index3D::Zero();
  while (raycaster.nextRayIndex(&ray_index)) {
    setIndexUpdated(ray_index, aabb_min, aabb_size, aabb_updated);
  }
}

template <typename SensorType>
std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycastTemplate(
    const DepthImage& depth_frame, const Transform& T_L_C,
    const SensorType& camera, const float block_size,
    const float truncation_distance_m, const float max_integration_distance_m) {
  timing::Timer setup_timer("in_view/setup");

  // Aight so first we have to get the AABB of this guy.
  const AxisAlignedBoundingBox aabb_L =
      camera.getViewAABB(T_L_C, 0.0f, max_integration_distance_m);

  // Get the min index and the max index.
  const Index3D min_index =
      getBlockIndexFromPositionInLayer(block_size, aabb_L.min());
  const Index3D max_index =
      getBlockIndexFromPositionInLayer(block_size, aabb_L.max());
  const Index3D aabb_size = max_index - min_index + Index3D::Ones();
  const size_t aabb_linear_size = aabb_size.x() * aabb_size.y() * aabb_size.z();

  // A 3D grid of bools, one for each block in the
  // AABB, which indicates if it is in the view. The 3D grid is represented
  // as a flat vector.
  if (aabb_linear_size > aabb_device_buffer_.size()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size =
        static_cast<int>(kBufferExpansionFactor * aabb_linear_size);
    aabb_device_buffer_.reserve(new_size);
    aabb_host_buffer_.reserve(new_size);
  }

  checkCudaErrors(cudaMemsetAsync(aabb_device_buffer_.data(), 0,
                                  sizeof(bool) * aabb_linear_size));
  aabb_device_buffer_.resize(aabb_linear_size);
  aabb_host_buffer_.resize(aabb_linear_size);

  setup_timer.Stop();

  // Raycast
  // default: true
  if (raycast_to_pixels_) {
    getBlocksByRaycastingPixels(T_L_C, camera, depth_frame, block_size,
                                truncation_distance_m,
                                max_integration_distance_m, min_index,
                                aabb_size, aabb_device_buffer_.data());
  } else {
    getBlocksByRaycastingCorners(T_L_C, camera, depth_frame, block_size,
                                 truncation_distance_m,
                                 max_integration_distance_m, min_index,
                                 aabb_size, aabb_device_buffer_.data());
  }

  // Output vector.
  timing::Timer output_timer("in_view/output");
  cudaMemcpyAsync(aabb_host_buffer_.data(), aabb_device_buffer_.data(),
                  sizeof(bool) * aabb_linear_size, cudaMemcpyDeviceToHost,
                  cuda_stream_);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  std::vector<Index3D> output_vector;
  convertAabbUpdatedToVector<std::vector<Index3D>>(
      min_index, aabb_size, aabb_linear_size, aabb_host_buffer_.data(),
      &output_vector);
  output_timer.Stop();

  // We have to manually destruct this. :(
  timing::Timer destory_timer("in_view/destroy");
  destory_timer.Stop();
  return output_vector;
}

// Camera
template <typename CameraType>
std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycast(
    const DepthImage& depth_frame, const Transform& T_L_C,
    const CameraType& camera, const float block_size,
    const float truncation_distance_m, const float max_integration_distance_m) {
  return getBlocksInImageViewRaycastTemplate(depth_frame, T_L_C, camera,
                                             block_size, truncation_distance_m,
                                             max_integration_distance_m);
}

// Lidar
std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycast(
    const DepthImage& depth_frame, const Transform& T_L_C, const Lidar& lidar,
    const float block_size, const float truncation_distance_m,
    const float max_integration_distance_m) {
  return getBlocksInImageViewRaycastTemplate(depth_frame, T_L_C, lidar,
                                             block_size, truncation_distance_m,
                                             max_integration_distance_m);
}

// OSLidar
std::vector<Index3D> ViewCalculator::getBlocksInImageViewRaycast(
    const DepthImage& depth_frame, const Transform& T_L_C, const OSLidar& lidar,
    const float block_size, const float truncation_distance_m,
    const float max_integration_distance_m) {
  return getBlocksInImageViewRaycastTemplate(depth_frame, T_L_C, lidar,
                                             block_size, truncation_distance_m,
                                             max_integration_distance_m);
}

template <typename SensorType>
void ViewCalculator::getBlocksByRaycastingCorners(
    const Transform& T_L_C, const SensorType& camera,
    const DepthImage& depth_frame, float block_size,
    const float truncation_distance_m, const float max_integration_distance_m,
    const Index3D& min_index, const Index3D& aabb_size,
    bool* aabb_updated_cuda) {
  // Get the blocks touched by the ray endpoints
  // We'll do warps of 32x32 pixels in the image. This is 1024 threads which is
  // in the recommended 512-1024 range.
  constexpr int kThreadDim = 16;
  int rounded_rows = static_cast<int>(
      std::ceil(depth_frame.rows() / static_cast<float>(kThreadDim)));
  int rounded_cols = static_cast<int>(
      std::ceil(depth_frame.cols() / static_cast<float>(kThreadDim)));
  dim3 block_dim(rounded_rows, rounded_cols);
  dim3 thread_dim(kThreadDim, kThreadDim);

  timing::Timer image_blocks_timer("in_view/get_image_blocks");
  getBlockIndicesInImageKernel<<<block_dim, thread_dim, 0, cuda_stream_>>>(
      T_L_C, camera, depth_frame.dataConstPtr(), depth_frame.rows(),
      depth_frame.cols(), block_size, max_integration_distance_m,
      truncation_distance_m, min_index, aabb_size, aabb_updated_cuda);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  image_blocks_timer.Stop();

  timing::Timer image_blocks_copy_timer("in_view/image_blocks_copy");

  unified_vector<Index3D> initial_vector;
  const size_t aabb_linear_size = aabb_size.x() * aabb_size.y() * aabb_size.z();
  initial_vector.reserve(aabb_linear_size / 3);
  convertAabbUpdatedToVector<unified_vector<Index3D>>(
      min_index, aabb_size, aabb_linear_size, aabb_updated_cuda,
      &initial_vector);
  image_blocks_copy_timer.Stop();

  // Call the kernel to do raycasting.
  timing::Timer raycast_blocks_timer("in_view/raycast_blocks");

  int num_initial_blocks = initial_vector.size();
  constexpr int kNumCorners = 9;
  constexpr int kNumBlocksPerThreadBlock = 40;
  int raycast_block_dim = static_cast<int>(std::ceil(
      static_cast<float>(num_initial_blocks) / kNumBlocksPerThreadBlock));
  dim3 raycast_thread_dim(kNumBlocksPerThreadBlock, kNumCorners);
  raycastToBlocksKernel<<<raycast_block_dim, raycast_thread_dim, 0,
                          cuda_stream_>>>(
      num_initial_blocks, initial_vector.data(), T_L_C, block_size, min_index,
      aabb_size, aabb_updated_cuda);
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());
  raycast_blocks_timer.Stop();
}

template <typename SensorType>
void ViewCalculator::getBlocksByRaycastingPixels(
    const Transform& T_L_C, const SensorType& camera,
    const DepthImage& depth_frame, float block_size,
    const float truncation_distance_m, const float max_integration_distance_m,
    const Index3D& min_index, const Index3D& aabb_size,
    bool* aabb_updated_cuda) {
  // Number of rays per dimension. Depth frame size / subsampling rate.
  const int num_subsampled_rows =
      std::ceil(static_cast<float>(depth_frame.rows() + 1) /
                static_cast<float>(raycast_subsampling_factor_));
  const int num_subsampled_cols =
      std::ceil(static_cast<float>(depth_frame.cols() + 1) /
                static_cast<float>(raycast_subsampling_factor_));

  // We'll do warps of 32x32 pixels in the image. This is 1024 threads which is
  // in the recommended 512-1024 range.
  constexpr int kThreadDim = 16;
  const int rounded_rows = static_cast<int>(
      std::ceil(num_subsampled_rows / static_cast<float>(kThreadDim)));
  const int rounded_cols = static_cast<int>(
      std::ceil(num_subsampled_cols / static_cast<float>(kThreadDim)));
  dim3 block_dim(rounded_rows, rounded_cols);
  dim3 thread_dim(kThreadDim, kThreadDim);

  timing::Timer combined_kernel_timer("in_view/combined_kernel");
  combinedBlockIndicesInImageKernel<<<block_dim, thread_dim, 0, cuda_stream_>>>(
      T_L_C, camera, depth_frame.dataConstPtr(), depth_frame.rows(),
      depth_frame.cols(), block_size, max_integration_distance_m,
      truncation_distance_m, raycast_subsampling_factor_, min_index, aabb_size,
      aabb_updated_cuda);

  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());
  combined_kernel_timer.Stop();
}

}  // namespace nvblox

// .cpp
// lidar.depth_frame_ptr[i] -> segmentation fault