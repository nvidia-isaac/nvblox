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
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>

#include "nvblox/gpu_hash/internal/cuda/gpu_hash_interface.cuh"
#include "nvblox/gpu_hash/internal/cuda/gpu_indexing.cuh"
#include "nvblox/rays/sphere_tracer.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

__device__ inline bool isTsdfVoxelValid(const TsdfVoxel& voxel) {
  constexpr float kMinWeight = 1e-4;
  return voxel.weight > kMinWeight;
}

__device__ thrust::pair<float, bool> cast(
    const Ray& ray,                                         // NOLINT
    const Index3DDeviceHashMapType<TsdfBlock>& block_hash,  // NOLINT
    float truncation_distance_m,                            // NOLINT
    float block_size_m,                                     // NOLINT
    int maximum_steps,                                      // NOLINT
    float maximum_ray_length_m,                             // NOLINT
    float surface_distance_epsilon_m) {
  // -------------------------------------------------------------------------
  // Approach: Step along the ray until we find the surface, or fail to find a
  // zero crossing.
  // -------------------------------------------------------------------------

  // The sign of the first valid distance we get.
  enum class FirstDistanceType { kPositive, kNegative, kNotYetKnown };
  FirstDistanceType first_valid_distance = FirstDistanceType::kNotYetKnown;

  // t captures the parameter scaling along ray.direction. We assume
  // that the ray is normalized which such that t has units meters.
  float t = 0.0f;
  for (int i = 0; (i < maximum_steps) && (t < maximum_ray_length_m); i++) {
    // Current point to sample
    const Vector3f p_L = ray.origin() + t * ray.direction();

    // Evaluate the distance at this point
    float step;
    TsdfVoxel* voxel_ptr;

    // Try to get a distance from the layer.
    // If we can't get a distance, let's see what to do...
    if (!getVoxelAtPosition(block_hash, p_L, block_size_m, &voxel_ptr) ||
        !isTsdfVoxelValid(*voxel_ptr)) {
      // 1) We weren't in observed space before this, let's step through this
      // (unobserved) shit and hope to hit something allocated.
      if (first_valid_distance == FirstDistanceType::kNotYetKnown) {
        // step forward by the truncation distance
        step = truncation_distance_m;
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
      // If this is our first sample in observed space
      if (first_valid_distance == FirstDistanceType::kNotYetKnown) {
        if (voxel_ptr->distance >= 0.0f) {
          first_valid_distance = FirstDistanceType::kPositive;
        } else {
          first_valid_distance = FirstDistanceType::kNegative;
        }
      }

      // If we're looking for Positive->Negative crossing
      if (first_valid_distance == FirstDistanceType::kPositive) {
        // If Distance negative (or close to it)!
        if (voxel_ptr->distance < surface_distance_epsilon_m) {
          // We found a zero crossing. Terminate successfully.
          // We're gonna terminate.
          // First we "refine" the distance by back stepping the (now negative)
          // distance value
          t += voxel_ptr->distance;
          // Output - Success!
          return {t, true};
        }
        // Distance positive! - keep searching
        else {
          // Step by this amount
          step = voxel_ptr->distance;
        }
      }
      // If we're looking for Negative->Positive crossing
      else {  // (first_valid_distance == FirstDistanceType::kNegative)
        // If Distance positive (or close to it)!
        if (voxel_ptr->distance > -surface_distance_epsilon_m) {
          // We "refine" the distance by back stepping the (now positive])
          // distance value
          t -= voxel_ptr->distance;
          // Output - Success!
          return {t, true};
        }
        // Distance Negative
        else {
          // Step by this amount
          step = -voxel_ptr->distance;
        }
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

__global__ void sphereTracingKernel(
    const Ray* rays_L,                               // NOLINT
    const int num_rays,                              // NOLINT
    Index3DDeviceHashMapType<TsdfBlock> block_hash,  // NOLINT
    Vector3f* points_L,                              // NOLINT
    bool* success_flags,                             // NOLINT
    float truncation_distance_m,                     // NOLINT
    float block_size_m,                              // NOLINT
    int maximum_steps,                               // NOLINT
    float maximum_ray_length_m,                      // NOLINT
    float surface_distance_epsilon_m) {
  // Extract a ray
  // NOTE(alexmillane): We expect this kernel to be called with sufficient
  // threads.
  const int ray_idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (ray_idx >= num_rays) {
    return;
  }
  const Ray ray_L = rays_L[ray_idx];

  // Cast
  const thrust::pair<float, bool> result =
      cast(ray_L, block_hash, truncation_distance_m, block_size_m,
           maximum_steps, maximum_ray_length_m, surface_distance_epsilon_m);

  // Reconstruct the 3D point
  if (!result.second) {
    success_flags[ray_idx] = result.second;
    return;
  }
  const Vector3f p_L = ray_L.pointAt(result.first);

  // Write the output
  points_L[ray_idx] = p_L;
  success_flags[ray_idx] = result.second;
}

__global__ void sphereTracingKernel(
    const Camera camera,                             // NOLINT
    const Transform T_L_C,                           // NOLINT
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
      camera.vectorFromImagePlaneCoordinates(pixel_coords).normalized();
  const Ray ray_L(T_L_C.translation(), T_L_C.linear() * ray_direction_C);

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

__global__ void sphereTracingKernelWithColor(
    const Camera camera,                                    // NOLINT
    const Transform T_L_C,                                  // NOLINT
    Index3DDeviceHashMapType<TsdfBlock> tsdf_block_hash,    // NOLINT
    Index3DDeviceHashMapType<ColorBlock> color_block_hash,  // NOLINT
    float* depth_image,                                     // NOLINT
    Color* color_image,
    float truncation_distance_m,       // NOLINT
    float block_size_m,                // NOLINT
    int maximum_steps,                 // NOLINT
    float maximum_ray_length_m,        // NOLINT
    float surface_distance_epsilon_m,  // NOLINT
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
      camera.vectorFromImagePlaneCoordinates(pixel_coords).normalized();
  const Ray ray_L(T_L_C.translation(), T_L_C.linear() * ray_direction_C);

  // Cast the ray into the TSDF layer
  thrust::pair<float, bool> t_optional =
      cast(ray_L, tsdf_block_hash, truncation_distance_m, block_size_m,
           maximum_steps, maximum_ray_length_m, surface_distance_epsilon_m);

  // If success, write depth to image, and look up the color. Otherwise write
  // -1, and a black color.
  if (t_optional.second == true) {
    const float depth = t_optional.first * ray_direction_C.z();
    image::access(ray_row_idx, ray_col_idx, ray_cols, depth_image) = depth;

    // Look up the color at the position where the ray bounced
    ColorVoxel* voxel_ptr;
    const Vector3f p_L = ray_L.pointAt(t_optional.first);
    if (getVoxelAtPosition(color_block_hash, p_L, block_size_m, &voxel_ptr)) {
      image::access(ray_row_idx, ray_col_idx, ray_cols, color_image) =
          voxel_ptr->color;
    } else {
      image::access(ray_row_idx, ray_col_idx, ray_cols, color_image) =
          Color(0, 0, 0);
    }

  } else {
    image::access(ray_row_idx, ray_col_idx, ray_cols, depth_image) = -1.0f;
    image::access(ray_row_idx, ray_col_idx, ray_cols, color_image) =
        Color(0, 0, 0);
  }
}

SphereTracer::SphereTracer()
    : SphereTracer(std::make_shared<CudaStreamOwning>()) {}

SphereTracer::SphereTracer(std::shared_ptr<CudaStream> cuda_stream)
    : cuda_stream_(cuda_stream) {}

int SphereTracer::maximum_steps() const { return maximum_steps_; }

float SphereTracer::maximum_ray_length_m() const {
  return maximum_ray_length_m_;
}

float SphereTracer::surface_distance_epsilon_vox() const {
  return surface_distance_epsilon_vox_;
}

void SphereTracer::maximum_steps(int maximum_steps) {
  CHECK_GT(maximum_steps, 0);
  maximum_steps_ = maximum_steps;
}

void SphereTracer::maximum_ray_length_m(float maximum_ray_length_m) {
  CHECK_GT(maximum_ray_length_m, 0);
  maximum_ray_length_m_ = maximum_ray_length_m;
}

void SphereTracer::surface_distance_epsilon_vox(
    float surface_distance_epsilon_vox) {
  CHECK_GT(surface_distance_epsilon_vox, 0);
  surface_distance_epsilon_vox_ = surface_distance_epsilon_vox;
}

bool SphereTracer::castOnGPU(const Ray& ray, const TsdfLayer& tsdf_layer,
                             const float truncation_distance_m,
                             float* t) const {
  constexpr float eps = 1e-5;
  CHECK_NEAR(ray.direction().norm(), 1.0, eps);

  // Get the GPU hash
  GPULayerView<TsdfBlock> gpu_layer_view = tsdf_layer.getGpuLayerView();

  // Allocate space
  float* t_device;
  bool* success_flag_device;
  checkCudaErrors(cudaMalloc(&t_device, sizeof(float)));
  checkCudaErrors(cudaMalloc(&success_flag_device, sizeof(bool)));

  // Kernel
  const float surface_distance_epsilon_m =
      surface_distance_epsilon_vox_ * tsdf_layer.voxel_size();
  sphereTracingKernel<<<1, 1, 0, *cuda_stream_>>>(
      ray,                             // NOLINT
      gpu_layer_view.getHash().impl_,  // NOLINT
      t_device,                        // NOLINT
      success_flag_device,             // NOLINT
      truncation_distance_m,           // NOLINT
      gpu_layer_view.block_size(),     // NOLINT
      maximum_steps_,                  // NOLINT
      maximum_ray_length_m_,           // NOLINT
      surface_distance_epsilon_m);

  // GPU -> CPU
  checkCudaErrors(cudaMemcpyAsync(t, t_device, sizeof(float),
                                  cudaMemcpyDeviceToHost, *cuda_stream_));
  bool success_flag;
  checkCudaErrors(cudaMemcpyAsync(&success_flag, success_flag_device,
                                  sizeof(bool), cudaMemcpyDeviceToHost,
                                  *cuda_stream_));

  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  // Deallocate
  checkCudaErrors(cudaFree(t_device));
  checkCudaErrors(cudaFree(success_flag_device));

  return success_flag;
}

void SphereTracer::renderImageOnGPU(const Camera& camera,
                                    const Transform& T_L_C,
                                    const TsdfLayer& tsdf_layer,
                                    const float truncation_distance_m,
                                    DepthImage* depth_ptr,
                                    const MemoryType output_image_memory_type,
                                    const int ray_subsampling_factor) {
  CHECK_NOTNULL(depth_ptr);

  CHECK_EQ(camera.width() % ray_subsampling_factor, 0);
  CHECK_EQ(camera.height() % ray_subsampling_factor, 0);
  CHECK(output_image_memory_type != MemoryType::kHost);
  // Output space
  const int image_height = camera.height() / ray_subsampling_factor;
  const int image_width = camera.width() / ray_subsampling_factor;

  // If we get a request for a different size image, reallocate.
  if (depth_ptr->width() != image_width ||
      depth_ptr->height() != image_height ||
      depth_ptr->memory_type() != output_image_memory_type) {
    LOG(INFO) << "Allocating output space for sphere tracing";
    *depth_ptr =
        DepthImage(image_height, image_width, output_image_memory_type);
  }

  DepthImageView depth_view(*depth_ptr);
  renderImageOnGPU(camera, T_L_C, tsdf_layer, truncation_distance_m,
                   &depth_view, output_image_memory_type,
                   ray_subsampling_factor);
}

void SphereTracer::renderImageOnGPU(const Camera& camera,
                                    const Transform& T_L_C,
                                    const TsdfLayer& tsdf_layer,
                                    const float truncation_distance_m,
                                    DepthImageView* depth_ptr,
                                    const MemoryType output_image_memory_type,
                                    const int ray_subsampling_factor) {
  CHECK_NOTNULL(depth_ptr);
  CHECK_NOTNULL(depth_ptr->dataPtr());

  CHECK_EQ(camera.width() % ray_subsampling_factor, 0);
  CHECK_EQ(camera.height() % ray_subsampling_factor, 0);
  CHECK(output_image_memory_type != MemoryType::kHost);
  // Output space
  const int image_height = camera.height() / ray_subsampling_factor;
  const int image_width = camera.width() / ray_subsampling_factor;

  // We can't allocate in this function so we check that the output is the right
  // size
  if (depth_ptr->cols() != image_width || depth_ptr->rows() != image_height) {
    LOG(WARNING) << "Attemped to sphere trace with incorrectly sized output "
                    "images. Doing nothings";
    return;
  }
  CHECK_EQ(depth_ptr->rows(), image_height);
  CHECK_EQ(depth_ptr->cols(), image_width);

  // Get the GPU hash
  timing::Timer hash_transfer_timer(
      "color/integrate/sphere_trace/hash_transfer");
  GPULayerView<TsdfBlock> gpu_layer_view = tsdf_layer.getGpuLayerView();
  hash_transfer_timer.Stop();

  // Get metric surface distance epsilon
  const float surface_distance_epsilon_m =
      surface_distance_epsilon_vox_ * tsdf_layer.voxel_size();

  // Kernel
  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(
      depth_ptr->cols() / kThreadsPerThreadBlock.x + 1,  // NOLINT
      depth_ptr->rows() / kThreadsPerThreadBlock.y + 1,  // NOLINT
      1);
  sphereTracingKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                        *cuda_stream_>>>(
      camera,                          // NOLINT
      T_L_C,                           // NOLINT
      gpu_layer_view.getHash().impl_,  // NOLINT
      depth_ptr->dataPtr(),            // NOLINT
      truncation_distance_m,           // NOLINT
      gpu_layer_view.block_size(),     // NOLINT
      maximum_steps_,                  // NOLINT
      maximum_ray_length_m_,           // NOLINT
      surface_distance_epsilon_m,      // NOLINT
      ray_subsampling_factor);
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

void SphereTracer::renderRgbdImageOnGPU(
    const Camera& camera, const Transform& T_L_C, const TsdfLayer& tsdf_layer,
    const ColorLayer& color_layer, const float truncation_distance_m,
    DepthImageView* depth_ptr, ColorImageView* color_ptr,
    const MemoryType output_image_memory_type,
    const int ray_subsampling_factor) {
  CHECK_EQ(camera.width() % ray_subsampling_factor, 0);
  CHECK_EQ(camera.height() % ray_subsampling_factor, 0);
  CHECK(output_image_memory_type != MemoryType::kHost);
  // Output space
  const int image_height = camera.height() / ray_subsampling_factor;
  const int image_width = camera.width() / ray_subsampling_factor;

  // We can't allocate in this function so we check that the output is the right
  // size. If not return
  if (depth_ptr->cols() != image_width || depth_ptr->rows() != image_height ||
      color_ptr->cols() != image_width || color_ptr->rows() != image_height) {
    LOG(WARNING) << "Attemped to sphere trace with incorrectly sized output "
                    "images. Doing nothings";
    return;
  }
  CHECK_EQ(depth_ptr->rows(), image_height);
  CHECK_EQ(depth_ptr->cols(), image_width);
  CHECK_EQ(color_ptr->rows(), image_height);
  CHECK_EQ(color_ptr->cols(), image_width);

  // Get the GPU hash
  timing::Timer hash_transfer_timer(
      "color/integrate/sphere_trace/hash_transfer");
  GPULayerView<TsdfBlock> tsdf_gpu_layer_view = tsdf_layer.getGpuLayerView();
  GPULayerView<ColorBlock> color_gpu_layer_view = color_layer.getGpuLayerView();
  hash_transfer_timer.Stop();

  // Get metric surface distance epsilon
  const float surface_distance_epsilon_m =
      surface_distance_epsilon_vox_ * tsdf_layer.voxel_size();

  // Kernel
  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(image_width / kThreadsPerThreadBlock.y + 1,   // NOLINT
                        image_height / kThreadsPerThreadBlock.x + 1,  // NOLINT
                        1);
  sphereTracingKernelWithColor<<<num_blocks, kThreadsPerThreadBlock, 0,
                                 *cuda_stream_>>>(
      camera,                                // NOLINT
      T_L_C,                                 // NOLINT
      tsdf_gpu_layer_view.getHash().impl_,   // NOLINT
      color_gpu_layer_view.getHash().impl_,  // NOLINT
      depth_ptr->dataPtr(),                  // NOLINT
      color_ptr->dataPtr(),                  // NOLINT
      truncation_distance_m,                 // NOLINT
      tsdf_gpu_layer_view.block_size(),      // NOLINT
      maximum_steps_,                        // NOLINT
      maximum_ray_length_m_,                 // NOLINT
      surface_distance_epsilon_m,            // NOLINT
      ray_subsampling_factor);
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());
}

void SphereTracer::renderRgbdImageOnGPU(
    const Camera& camera, const Transform& T_L_C, const TsdfLayer& tsdf_layer,
    const ColorLayer& color_layer, const float truncation_distance_m,
    DepthImage* depth_ptr, ColorImage* color_ptr,
    const MemoryType output_image_memory_type,
    const int ray_subsampling_factor) {
  CHECK_NOTNULL(depth_ptr);
  CHECK_NOTNULL(color_ptr);
  CHECK_EQ(camera.width() % ray_subsampling_factor, 0);
  CHECK_EQ(camera.height() % ray_subsampling_factor, 0);
  CHECK(output_image_memory_type != MemoryType::kHost);
  // Output space
  const int image_height = camera.height() / ray_subsampling_factor;
  const int image_width = camera.width() / ray_subsampling_factor;

  // If we get a request for a different size image, reallocate.
  if (depth_ptr->width() != image_width ||
      depth_ptr->height() != image_height ||
      depth_ptr->memory_type() != output_image_memory_type) {
    LOG(INFO) << "Allocating output space for sphere tracing";
    *depth_ptr =
        DepthImage(image_height, image_width, output_image_memory_type);
  }
  if (color_ptr->width() != image_width ||
      color_ptr->height() != image_height ||
      color_ptr->memory_type() != output_image_memory_type) {
    LOG(INFO) << "Allocating output space for sphere tracing";
    *color_ptr =
        ColorImage(image_height, image_width, output_image_memory_type);
  }

  DepthImageView depth_image_view(*depth_ptr);
  ColorImageView color_image_view(*color_ptr);
  renderRgbdImageOnGPU(camera, T_L_C, tsdf_layer, color_layer,
                       truncation_distance_m, &depth_image_view,
                       &color_image_view, output_image_memory_type,
                       ray_subsampling_factor);
}

std::pair<device_vector<Vector3f>, device_vector<bool>> SphereTracer::castOnGPU(
    std::vector<Ray>& rays_L, const TsdfLayer& tsdf_layer,
    const float truncation_distance_m) {
  // Inputs
  device_vector<Ray> rays_L_device;
  rays_L_device.copyFromAsync(rays_L, *cuda_stream_);

  // Output space
  device_vector<Vector3f> intersection_points_L(rays_L.size());
  device_vector<bool> success_flags(rays_L.size());
  success_flags.setZero();

  // Get the GPU hash
  timing::Timer hash_transfer_timer(
      "color/integrate/sphere_trace/hash_transfer");
  GPULayerView<TsdfBlock> gpu_layer_view = tsdf_layer.getGpuLayerView();
  hash_transfer_timer.Stop();

  // Get metric surface distance epsilon
  const float surface_distance_epsilon_m =
      surface_distance_epsilon_vox_ * tsdf_layer.voxel_size();

  // Kernel
  // Call params
  // - 1 thread per pixel
  // - 64 threads per thread block (chosen arbitrarily)
  // - N thread blocks get 1 thread per ray
  constexpr int kThreadsPerThreadBlock = 32;
  const int num_blocks = rays_L.size() / kThreadsPerThreadBlock + 1;
  sphereTracingKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                        *cuda_stream_>>>(
      rays_L_device.data(),            // NOLINT
      rays_L_device.size(),            // NOLINT
      gpu_layer_view.getHash().impl_,  // NOLINT
      intersection_points_L.data(),    // NOLINT
      success_flags.data(),            // NOLINT
      truncation_distance_m,           // NOLINT
      gpu_layer_view.block_size(),     // NOLINT
      maximum_steps_,                  // NOLINT
      maximum_ray_length_m_,           // NOLINT
      surface_distance_epsilon_m);
  cuda_stream_->synchronize();
  checkCudaErrors(cudaPeekAtLastError());

  return {std::move(intersection_points_L), std::move(success_flags)};
}

}  // namespace nvblox
