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
#include "nvblox/core/common_names.h"
#include "nvblox/core/image.h"
#include "nvblox/gpu_hash/gpu_layer_view.h"

namespace nvblox {

struct Ray {
  Vector3f direction;
  Vector3f origin;
};

class SphereTracer {
 public:
  struct Params {
    Params() {}

    int maximum_steps = 100;
    float maximum_ray_length_m = 15.0f;
    float surface_distance_epsilon_vox = 0.1f;
  };

  SphereTracer(Params params = Params());
  ~SphereTracer();

  // Render an image
  // We return a pointer to the buffer on GPU where the image is rendered.
  // Note that additional calls to this function will change the contents of the
  // pointed-to image.
  std::shared_ptr<const DepthImage> renderImageOnGPU(
      const Camera& camera, const Transform& T_S_C, const TsdfLayer& tsdf_layer,
      const float truncation_distance_m,
      const MemoryType output_image_memory_type = MemoryType::kDevice,
      const int ray_subsampling_factor = 1);

  // NOTE(alexmillane): This starts a single threaded kernel to cast the ray
  // and so is not efficient. Intended for testing rather than production.
  bool castOnGPU(const Ray& ray, const TsdfLayer& tsdf_layer,
                 const float truncation_distance_m, float* t) const;

  const Params& params() const { return params_; }
  Params& params() { return params_; }

 private:
  // Device working space
  float* t_device_;
  bool* success_flag_device_;
  std::shared_ptr<DepthImage> depth_image_;

  Params params_;
  cudaStream_t tracing_stream_;
};

}  // namespace nvblox
