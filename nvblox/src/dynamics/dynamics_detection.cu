/*
Copyright 2025 NVIDIA CORPORATION

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

namespace nvblox {

DynamicsDetection::DynamicsDetection(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

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

  // Images
  dynamics_mask_.resizeAsync(rows, cols, *cuda_stream_);
  dynamics_overlay_.resizeAsync(rows, cols, *cuda_stream_);

  // Point counters
  if (dynamic_points_counter_device_ == nullptr ||
      dynamic_points_counter_host_ == nullptr) {
    dynamic_points_counter_device_ = make_unified<int>(MemoryType::kDevice);
    dynamic_points_counter_host_ = make_unified<int>(MemoryType::kHost);
  }
  dynamic_points_counter_device_.setZeroAsync(*cuda_stream_);

  // Points
  if (static_cast<size_t>(num_input_pixels) > dynamic_points_device_.size()) {
    dynamic_points_device_.resizeAsync(num_input_pixels, *cuda_stream_);
  }
}

}  // namespace nvblox
