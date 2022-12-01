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
#pragma once

#include <memory>
#include <utility>

#include "nvblox/core/camera.h"
#include "nvblox/core/camera_pinhole.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/image.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"

namespace nvblox {

/// A class for rendering synthetic depth images using sphere tracing.
class SphereTracer {
 public:
  SphereTracer();
  ~SphereTracer();

  /// Render an image on the GPU
  /// Rendering occurs by sphere tracing the passed TsdfLayer. We return a
  /// pointer to an internal GPU buffer where the image is rendered. Note that
  /// additional calls to this function will change the contents of the
  /// internal image buffer and therefore invalidate the returned image.
  /// @param camera A the camera (intrinsics) model.
  /// @param T_L_C The pose of the camera. Supplied as a Transform mapping
  /// points in the camera frame (C) to the layer frame (L).
  /// @param tsdf_layer The tsdf layer to be sphere traced.
  /// @param truncation_distance_m The (metric) truncation distance used during
  /// the construction of tsdf_layer.
  /// @param output_image_memory_type The memory type that the image should be
  /// stored in.
  /// @param ray_subsampling_factor The subsampling rate applied to the number
  /// of traced rays. If this parameter is 2 for example and the image is
  /// 100x100 pixels, we trace 50x50 pixels and return a syntheric depth image
  /// of that size.
  /// @returns A pointer to the internal (GPU) buffer where the image is stored.
  template <typename CameraType>
  std::shared_ptr<const DepthImage> renderImageOnGPU(
      const CameraType& camera, const Transform& T_L_C,
      const TsdfLayer& tsdf_layer, const float truncation_distance_m,
      const MemoryType output_image_memory_type = MemoryType::kDevice,
      const int ray_subsampling_factor = 1);

  /// Sphere trace a ray-bundle on the GPU
  /// We leave the results in buffers on the GPU.
  /// @param rays_L a vector of rays in the Layer frame
  /// @param tsdf_layer the tsdf layer to sphere traced
  /// @param truncation_distance_m The (metric) truncation distance used during
  /// the construction of tsdf_layer.
  std::pair<device_vector<Vector3f>, device_vector<bool>> castOnGPU(
      std::vector<Ray>& rays_L, const TsdfLayer& tsdf_layer,
      const float truncation_distance_m);

  // This starts a single threaded kernel to cast a single ray
  // and is therefore not efficient. This function is intended for unit testing.
  bool castOnGPU(const Ray& ray, const TsdfLayer& tsdf_layer,
                 const float truncation_distance_m, float* t) const;

  /// A parameter getter.
  /// The maximum number of steps along a ray allowed before ray casting fails.
  /// @returns the maximum number of steps
  int maximum_steps() const;

  /// A parameter getter.
  /// The maximum distance along a ray allowed before ray casting fails.
  /// @returns the maximum length
  float maximum_ray_length_m() const;

  /// A parameter getter.
  /// The distance in voxels, sampled from the underlying distance field, at
  /// which under which sphere casting determines that it has successfully found
  /// a surface.
  /// @returns the distance to the surface.
  float surface_distance_epsilon_vox() const;

  /// A parameter setter.
  /// See maximum_steps().
  /// @param maximum_steps the maximum number of steps along the ray.
  void maximum_steps(int maximum_steps);

  /// A parameter setter.
  /// See maximum_ray_length_m().
  /// @param maximum_ray_length_m the maximum distance along the ray.
  void maximum_ray_length_m(float maximum_ray_length_m);

  /// A parameter setter.
  /// See surface_distance_epsilon_vox().
  /// @param surface_distance_epsilon_vox the distance to the surface.
  void surface_distance_epsilon_vox(float surface_distance_epsilon_vox);

 protected:
  // Device working space
  std::shared_ptr<DepthImage> depth_image_;

  // Params
  int maximum_steps_ = 100;
  float maximum_ray_length_m_ = 15.0f;
  float surface_distance_epsilon_vox_ = 0.1f;

  // The CUDA stream on which processing occurs
  cudaStream_t tracing_stream_;
};

}  // namespace nvblox
