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
#include "nvblox/dynamics/dynamics_detection.h"

#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"

namespace nvblox {

__device__ Color getOverlayColor(const bool is_dynamic, const float depth) {
  constexpr float max_display_depth_m = 10.f;
  constexpr float depth_scale_factor = 255.0f / max_display_depth_m;
  const uint8_t scaled_depth = fmin(depth_scale_factor * depth, 255u);

  // Dynamics shown in red and rest greyish scaled depending on depth
  return Color(is_dynamic * 255u, scaled_depth, scaled_depth);
}

__global__ void findDynamicPointsKernel(
    const float* depth_frame_C,
    const Index3DDeviceHashMapType<FreespaceBlock> block_hash, float block_size,
    const Transform T_L_C, const Camera camera, const int rows, const int cols,
    int* dynamic_points_counter, Vector3f* dynamic_points,
    uint8_t* dynamic_mask_image, Color* dynamic_overlay_image) {
  // Each thread does a single pixel on the depth image.
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }

  // Set pixel default values on output images.
  image::access(row_idx, col_idx, cols, dynamic_mask_image) = 0;
  image::access(row_idx, col_idx, cols, dynamic_overlay_image) = Color::White();

  // Get depth value.
  const float depth = image::access(row_idx, col_idx, cols, depth_frame_C);
  if (depth <= 0.0f) {
    return;  // Depth pixel invalid
  }

  // Get 3D point in the freespace layer frame.
  const Vector3f point_C =
      camera.unprojectFromPixelIndices(Index2D(col_idx, row_idx), depth);
  const Vector3f point_L = T_L_C * point_C;

  // Get the corresponding voxel.
  FreespaceVoxel* freespace_voxel;
  if (!getVoxelAtPosition<FreespaceVoxel>(block_hash, point_L, block_size,
                                          &freespace_voxel)) {
    return;  // Voxel not found.
  }

  // If a projected depth pixel falls into a high confidence freespace voxel we
  // assume it must be dynamic.
  const bool is_dynamic = freespace_voxel->is_high_confidence_freespace;

  // Store dynamic points.
  if (is_dynamic) {
    int current_idx = atomicAdd(dynamic_points_counter, 1);
    dynamic_points[current_idx] = point_L;
  }

  // Update mask and overlay image
  image::access(row_idx, col_idx, cols, dynamic_mask_image) = is_dynamic;
  image::access(row_idx, col_idx, cols, dynamic_overlay_image) =
      getOverlayColor(is_dynamic, depth);
}

DynamicsDetection::DynamicsDetection()
    : cuda_stream_(std::make_shared<CudaStreamOwning>()) {}

DynamicsDetection::DynamicsDetection(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

void DynamicsDetection::computeDynamics(const DepthImage& depth_frame_C,
                                        const FreespaceLayer& freespace_layer_L,
                                        const Camera& camera,
                                        const Transform& T_L_C) {
  const int rows = depth_frame_C.rows();
  const int cols = depth_frame_C.cols();
  prepareOutputs(depth_frame_C);

  // Kernel call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(cols / kThreadsPerThreadBlock.x + 1,
                        rows / kThreadsPerThreadBlock.y + 1, 1);
  findDynamicPointsKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                            *cuda_stream_>>>(
      depth_frame_C.dataConstPtr(),                         // NOLINT
      freespace_layer_L.getGpuLayerView().getHash().impl_,  // NOLINT
      freespace_layer_L.block_size(),                       // NOLINT
      T_L_C,                                                // NOLINT
      camera,                                               // NOLINT
      rows,                                                 // NOLINT
      cols,                                                 // NOLINT
      dynamic_points_counter_device_.get(),                 // NOLINT
      dynamic_points_device_.data(),                        // NOLINT
      dynamics_mask_.dataPtr(),                             // NOLINT
      dynamics_overlay_.dataPtr());                         // NOLINT
  dynamic_points_counter_device_.copyToAsync(dynamic_points_counter_host_,
                                             *cuda_stream_);
  cuda_stream_->synchronize();
  dynamic_points_device_.resizeAsync(*dynamic_points_counter_host_,
                                     *cuda_stream_);
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

Eigen::Matrix3Xf DynamicsDetection::getDynamicPointsHost() {
  // Copy to host.
  dynamic_points_host_.copyFromAsync(dynamic_points_device_, *cuda_stream_);
  cuda_stream_->synchronize();

  // Convert to eigen.
  return Eigen::Matrix3Xf::Map(dynamic_points_host_.data()->data(), 3,
                               *dynamic_points_counter_host_);
}

const Pointcloud& DynamicsDetection::getDynamicPointcloudDevice() {
  dynamic_pointcloud_device_.copyFromAsync(dynamic_points_device_,
                                           *cuda_stream_);
  cuda_stream_->synchronize();
  return dynamic_pointcloud_device_;
}

const MonoImage& DynamicsDetection::getDynamicMaskImage() const {
  return dynamics_mask_;
}

const ColorImage& DynamicsDetection::getDynamicOverlayImage() const {
  return dynamics_overlay_;
}

void DynamicsDetection::prepareOutputs(const DepthImage& input_frame) {
  CHECK(input_frame.memory_type() != MemoryType::kHost);

  // Get input sizes
  const int num_input_pixels = input_frame.numel();
  const int rows = input_frame.rows();
  const int cols = input_frame.cols();

  // Mask
  if ((dynamics_mask_.rows() != input_frame.rows()) ||
      (dynamics_mask_.cols() != input_frame.cols()) ||
      dynamics_mask_.memory_type() == input_frame.memory_type()) {
    dynamics_mask_ = MonoImage(rows, cols, input_frame.memory_type());
  }

  // Overlay
  if ((dynamics_overlay_.rows() != input_frame.rows()) ||
      (dynamics_overlay_.cols() != input_frame.cols()) ||
      dynamics_overlay_.memory_type() == input_frame.memory_type()) {
    dynamics_overlay_ = ColorImage(rows, cols, input_frame.memory_type());
  }

  // Point counters
  if (dynamic_points_counter_device_ == nullptr ||
      dynamic_points_counter_host_ == nullptr) {
    dynamic_points_counter_device_ = make_unified<int>(MemoryType::kDevice);
    dynamic_points_counter_host_ = make_unified<int>(MemoryType::kHost);
  }
  dynamic_points_counter_device_.setZero();

  // Points
  if (static_cast<size_t>(num_input_pixels) > dynamic_points_device_.size()) {
    dynamic_points_device_.resize(num_input_pixels);
  }
}

}  // namespace nvblox
