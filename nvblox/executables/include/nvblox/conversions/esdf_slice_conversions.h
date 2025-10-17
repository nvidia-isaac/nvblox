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

#pragma once

#include <nvblox/nvblox.h>
#include <memory>

namespace nvblox {
namespace conversions {

class EsdfSliceConverter {
 public:
  EsdfSliceConverter();
  explicit EsdfSliceConverter(std::shared_ptr<CudaStream> cuda_stream);

  // ------------- WRAPPING ESDF SLICER FUNCTIONS -------------

  // Slicing an esdf layer (using EsdfSlicer)
  void sliceLayerToDistanceImage(
    const EsdfLayer& layer, float slice_height, float unobserved_value,
    Image<float>* output_image,
    AxisAlignedBoundingBox* aabb);

  // Convert slice image to occupancy grid
  void occupancyGridFromSliceImage(
    const Image<float>& slice_image, signed char* occupancy_grid_data,
    float unknown_value);

 private:
  // Slicer that does the work
  EsdfSlicer esdf_slicer_;

  std::shared_ptr<CudaStream> cuda_stream_;

  // Buffers
  device_vector<int8_t> occupancy_grid_device_;
};

}  // namespace conversions
}  // namespace nvblox
