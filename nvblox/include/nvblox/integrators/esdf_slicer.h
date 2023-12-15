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
#pragma once

#include "nvblox/core/types.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

class EsdfSlicer {
 public:
  EsdfSlicer();
  EsdfSlicer(std::shared_ptr<CudaStream> cuda_stream);
  virtual ~EsdfSlicer() = default;

  /// Returns the AABB of an esdf layer at a specific height.
  /// @param layer Input ESDF layer.
  /// @param slice_height The height where the AABB should be generated.
  AxisAlignedBoundingBox getAabbOfLayerAtHeight(const EsdfLayer& layer,
                                                const float slice_height);

  /// Returns the combined AABB of two esdf layers at a specific height.
  /// @param layer_1 First input ESDF layer.
  /// @param layer_2 Second input ESDF layer.
  /// @param slice_height The height where the AABB should be generated.
  AxisAlignedBoundingBox getCombinedAabbOfLayersAtHeight(
      const EsdfLayer& layer_1, const EsdfLayer& layer_2,
      const float slice_height);

  /// Slices an ESDF layer at a specific height to a distance image inside a
  /// custom AABB.
  /// @param layer Input ESDF layer.
  /// @param slice_height The height of the slice to output.
  /// @param unobserved_value Floating-point value to use for unknown/unobserved
  /// points.
  /// @param aabb AABB to generate the distance image in. Used as-is; if it's
  /// larger than the layer, the rest is just filled in as unknown value.
  /// @param output_image Output floating point image with the distances at each
  /// pixel.
  void sliceLayerToDistanceImage(const EsdfLayer& layer, float slice_height,
                                 float unobserved_value,
                                 const AxisAlignedBoundingBox& aabb,
                                 Image<float>* output_image);

  /// Slices two ESDF layers at a specific height to a combined distance image
  /// inside a custom AABB.
  /// @param layer_1 First input ESDF layer.
  /// @param layer_2 Second input ESDF layer.
  /// @param slice_height The height of the slice to output.
  /// @param unobserved_value Floating-point value to use for unknown/unobserved
  /// points.
  /// @param aabb AABB to generate the distance image in. Used as-is; if it's
  /// larger than the layers, the rest is just filled in as unknown value.
  /// @param output_image Output floating point image with the distances at each
  /// pixel.
  void sliceLayersToCombinedDistanceImage(const EsdfLayer& layer_1,
                                          const EsdfLayer& layer_2,
                                          float slice_height,
                                          float unobserved_value,
                                          const AxisAlignedBoundingBox& aabb,
                                          Image<float>* output_image);

 private:
  // Helper to do the actual work.
  void populateSliceFromLayer(const EsdfLayer& layer,
                              const AxisAlignedBoundingBox& aabb,
                              float slice_height, float unobserved_value,
                              float resolution, Image<float>* output_image);

  std::shared_ptr<CudaStream> cuda_stream_;
};

}  // namespace nvblox
