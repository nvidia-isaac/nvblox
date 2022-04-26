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
#include "nvblox/rays/sphere_tracer.h"

#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>

#include "nvblox/gpu_hash/cuda/gpu_hash_interface.cuh"
#include "nvblox/gpu_hash/cuda/gpu_indexing.cuh"
#include "nvblox/utils/timing.h"

namespace nvblox {

__device__ inline bool isTsdfVoxelValid(const TsdfVoxel& voxel) {
  constexpr float kMinWeight = 1e-4;
  return voxel.weight > kMinWeight;
}

__device__ thrust::pair<float, bool> cast(
    const Ray& ray,                                  // NOLINT
    Index3DDeviceHashMapType<TsdfBlock> block_hash,  // NOLINT
    float truncation_distance_m,                     // NOLINT
    float block_size_m,                              // NOLINT
    int maximum_steps,                               // NOLINT
    float maximum_ray_length_m,                      // NOLINT
    float surface_distance_epsilon_m) {
  // Approach: Step along the ray until we find the surface, or fail to
  bool last_distance_positive = false;
  // t captures the parameter scaling along ray.direction. We assume
  // that the ray is normalized which such that t has units meters.
  float t = 0.0f;
  for (int i = 0; (i < maximum_steps) && (t < maximum_ray_length_m); i++) {
    // Current point to sample
    const Vector3f p_L = ray.origin + t * ray.direction;

    // Evaluate the distance at this point
    float step;
    TsdfVoxel* voxel_ptr;

    // Can't get distance, let's see what to do...
    if (!getVoxelAtPosition(block_hash, p_L, block_size_m, &voxel_ptr) ||
        !isTsdfVoxelValid(*voxel_ptr)) {
      // 1) We weren't in observed space before this, let's step through this
      // (unobserved) shit and hope to hit something allocated.
      if (!last_distance_positive) {
        // step forward by the truncation distance
        step = truncation_distance_m;
        last_distance_positive = false;
      }
      // 2) We were in observed space, now we've left it... let's kill this
      // ray, it's risky to continue.
      // Note(alexmillane): The "risk" here is that we've somehow passed
      // through the truncation band. This occurs occasionally. The risk
      // of continuing is that we can then see through an object. It's safer
      // to stop here and hope for better luck in the next frame.
      else {
        return {t, false};
      }
    }
    // We got a valid distance
    else {
      // Distance negative (or close to it)!
      // We're gonna terminate, let's determine how.
      if (voxel_ptr->distance < surface_distance_epsilon_m) {
        // 1) We found a zero crossing. Terminate successfully.
        if (last_distance_positive) {
          // We "refine" the distance by back stepping the (now negative)
          // distance value
          t += voxel_ptr->distance;
          // Output - Success!
          return {t, true};
        }
        // 2) We just went from unobserved to negative. We're observing
        // something from behind, terminate.
        else {
          return {t, false};
        }
      }
      // Distance positive!
      else {
        // Step by this amount
        step = voxel_ptr->distance;
        last_distance_positive = true;
      }
    }

    // Step further along the ray
    t += step;
  }
  // Ran out of number of steps or distance... Fail
  return {t, false};
}

__global__ void sphereTracingKernel(
    const Ray ray,                                   // NOLINT
    Index3DDeviceHashMapType<TsdfBlock> block_hash,  // NOLINT
    float* t,                                        // NOLINT
    bool* success_flag,                              // NOLINT
    float truncation_distance_m,                     // NOLINT
    float block_size_m,                              // NOLINT
    int maximum_steps,                               // NOLINT
    float maximum_ray_length_m,                      // NOLINT
    float surface_distance_epsilon_m) {
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx != 0) return;

  thrust::pair<float, bool> res =
      cast(ray, block_hash, truncation_distance_m, block_size_m, maximum_steps,
           maximum_ray_length_m, surface_distance_epsilon_m);

  *t = res.first;
  *success_flag = res.second;
}

__global__ void sphereTraceImageKernel(
    const Camera camera,                             // NOLINT
    const Transform T_S_C,                           // NOLINT
    Index3DDeviceHashMapType<TsdfBlock> block_hash,  // NOLINT
    float* image,                                    // NOLINT
    float truncation_distance_m,                     // NOLINT
    float block_size_m,                              // NOLINT
    int maximum_steps,                               // NOLINT
    float maximum_ray_length_m,                      // NOLINT
    float surface_distance_epsilon_m,                // NOLINT
    int ray_subsampling_factor) {
  const int ray_col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int ray_row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  // Note: we ensure that this division works cleanly before getting here.
  const int ray_rows = camera.rows() / ray_subsampling_factor;
  const int ray_cols = camera.cols() / ray_subsampling_factor;
  if ((ray_row_idx >= ray_rows) || (ray_col_idx >= ray_cols)) {
    return;
  }

  // Get the image-plane coordinates of where this ray should pass such that it
  // is in the center of the patch it will represent.
  constexpr float kHalf = 1.0f / 2.0f;
  const Index2D ray_indices(ray_col_idx, ray_row_idx);
  const Vector2f pixel_coords =
      (ray_indices * ray_subsampling_factor).cast<float>() +
      kHalf * static_cast<float>(ray_subsampling_factor) * Vector2f::Ones();

  // Get the ray going through this pixel (in layer coordinate)
  const Vector3f ray_direction_C =
      camera.rayFromImagePlaneCoordinates(pixel_coords).normalized();
  const Ray ray_L{T_S_C.linear() * ray_direction_C, T_S_C.translation()};

  // Cast the ray into the layer
  thrust::pair<float, bool> t_optional =
      cast(ray_L, block_hash, truncation_distance_m, block_size_m,
           maximum_steps, maximum_ray_length_m, surface_distance_epsilon_m);

  // If success, write depth to image, otherwise write -1.
  if (t_optional.second == true) {
    const float depth = t_optional.first * ray_direction_C.z();
    image::access(ray_row_idx, ray_col_idx, ray_cols, image) = depth;
  } else {
    image::access(ray_row_idx, ray_col_idx, ray_cols, image) = -1.0f;
  }
}

SphereTracer::SphereTracer(Params params) : params_(std::move(params)) {
  checkCudaErrors(cudaStreamCreate(&tracing_stream_));
  cudaMalloc(&t_device_, sizeof(float));
  cudaMalloc(&success_flag_device_, sizeof(bool));
}

SphereTracer::~SphereTracer() {
  cudaStreamSynchronize(tracing_stream_);
  cudaFree(t_device_);
  cudaFree(success_flag_device_);
  checkCudaErrors(cudaStreamDestroy(tracing_stream_));
}

bool SphereTracer::castOnGPU(const Ray& ray, const TsdfLayer& tsdf_layer,
                             const float truncation_distance_m,
                             float* t) const {
  constexpr float eps = 1e-5;
  CHECK_NEAR(ray.direction.norm(), 1.0, eps);

  // Get the GPU hash
  GPULayerView<TsdfBlock> gpu_layer_view = tsdf_layer.getGpuLayerView();

  // Kernel
  const float surface_distance_epsilon_m =
      params_.surface_distance_epsilon_vox * tsdf_layer.voxel_size();
  sphereTracingKernel<<<1, 1, 0, tracing_stream_>>>(
      ray,                             // NOLINT
      gpu_layer_view.getHash().impl_,  // NOLINT
      t_device_,                       // NOLINT
      success_flag_device_,            // NOLINT
      truncation_distance_m,           // NOLINT
      gpu_layer_view.block_size(),     // NOLINT
      params_.maximum_steps,           // NOLINT
      params_.maximum_ray_length_m,    // NOLINT
      surface_distance_epsilon_m);

  // GPU -> CPU
  cudaMemcpyAsync(t, t_device_, sizeof(float), cudaMemcpyDeviceToHost,
                  tracing_stream_);

  checkCudaErrors(cudaStreamSynchronize(tracing_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  return true;
}

std::shared_ptr<const DepthImage> SphereTracer::renderImageOnGPU(
    const Camera& camera, const Transform& T_S_C, const TsdfLayer& tsdf_layer,
    const float truncation_distance_m,
    const MemoryType output_image_memory_type,
    const int ray_subsampling_factor) {
  CHECK_EQ(camera.width() % ray_subsampling_factor, 0);
  CHECK_EQ(camera.height() % ray_subsampling_factor, 0);
  // Output space
  const int image_height = camera.height() / ray_subsampling_factor;
  const int image_width = camera.width() / ray_subsampling_factor;
  // If we get a request for a different size image, reallocate.
  if (!depth_image_ || depth_image_->width() != image_width ||
      depth_image_->height() != image_height ||
      depth_image_->memory_type() != output_image_memory_type) {
    depth_image_ = std::make_shared<DepthImage>(image_height, image_width,
                                                output_image_memory_type);
  }

  // Get the GPU hash
  timing::Timer hash_transfer_timer(
      "color/integrate/sphere_trace/hash_transfer");
  GPULayerView<TsdfBlock> gpu_layer_view = tsdf_layer.getGpuLayerView();
  hash_transfer_timer.Stop();

  // Get metric surface distance epsilon
  const float surface_distance_epsilon_m =
      params_.surface_distance_epsilon_vox * tsdf_layer.voxel_size();

  // Kernel
  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(
      depth_image_->cols() / kThreadsPerThreadBlock.y + 1,  // NOLINT
      depth_image_->rows() / kThreadsPerThreadBlock.x + 1,  // NOLINT
      1);
  sphereTraceImageKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                           tracing_stream_>>>(
      camera,                          // NOLINT
      T_S_C,                           // NOLINT
      gpu_layer_view.getHash().impl_,  // NOLINT
      depth_image_->dataPtr(),         // NOLINT
      truncation_distance_m,           // NOLINT
      gpu_layer_view.block_size(),     // NOLINT
      params_.maximum_steps,           // NOLINT
      params_.maximum_ray_length_m,    // NOLINT
      surface_distance_epsilon_m,      // NOLINT
      ray_subsampling_factor);
  checkCudaErrors(cudaStreamSynchronize(tracing_stream_));
  checkCudaErrors(cudaPeekAtLastError());

  return depth_image_;
}

}  // namespace nvblox
