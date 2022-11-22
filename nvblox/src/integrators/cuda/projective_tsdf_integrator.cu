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
#include <nvblox/integrators/projective_tsdf_integrator.h>

#include "nvblox/core/color.h"
#include "nvblox/core/cuda/error_check.cuh"
#include "nvblox/core/interpolation_2d.h"
#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/utils/timing.h"
#include "nvblox/utils/weight_function.h"

namespace nvblox {

// NOTE(jjiao): the original nvblox implementation
__device__ inline bool updateVoxel(const float surface_depth_measured,
                                   TsdfVoxel* voxel_ptr,
                                   const float voxel_depth_m,
                                   const float truncation_distance_m,
                                   const float max_weight) {
  // Get the MEASURED depth of the VOXEL
  float voxel_distance_measured = surface_depth_measured - voxel_depth_m;

  // If we're behind the negative truncation distance, just continue.
  if (voxel_distance_measured < -truncation_distance_m) {
    return false;
  }

  // Read CURRENT voxel values (from global GPU memory)
  const float voxel_distance_current = voxel_ptr->distance;
  const float voxel_weight_current = voxel_ptr->weight;

  // NOTE(alexmillane): We could try to use CUDA math functions to speed up
  // below
  // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE
  // Fuse
  constexpr float measurement_weight = 1.0f;
  float fused_distance = (voxel_distance_measured * measurement_weight +
                          voxel_distance_current * voxel_weight_current) /
                         (measurement_weight + voxel_weight_current);
  // Clip
  if (fused_distance > 0.0f) {
    fused_distance = fminf(truncation_distance_m, fused_distance);
  } else {
    fused_distance = fmaxf(-truncation_distance_m, fused_distance);
  }
  const float weight =
      fminf(measurement_weight + voxel_weight_current, max_weight);
  // Write NEW voxel values (to global GPU memory)
  voxel_ptr->distance = fused_distance;
  voxel_ptr->weight = weight;
  return true;
}

// TODO(jjiao): update the voxel according to the traditional TSDF update method
// TODO(jjiao): we can try different methods based on the probabilitics to
// improve the voxel update
__device__ inline bool updateVoxelMultiWeightComp(
    const float surface_depth_measured, TsdfVoxel* voxel_ptr,
    const float voxel_depth_m, const float truncation_distance_m,
    const float max_weight, const int voxel_dis_method,
    const Vector3f measurement_point, const Vector3f& measurement_normal,
    const Transform& T_C_L) {
  // Get the MEASURED depth of the VOXEL
  float voxel_distance_measured = surface_depth_measured - voxel_depth_m;
  // If we're behind the negative truncation distance, just continue.
  if (voxel_distance_measured < -truncation_distance_m) {
    return false;
  }

  // Read CURRENT voxel values (from global GPU memory)
  const float voxel_distance_current = voxel_ptr->distance;
  const float voxel_weight_current = voxel_ptr->weight;
  const Vector3f voxel_gradient_current = voxel_ptr->gradient;

  // NOTE(alexmillane): We could try to use CUDA math functions to speed up
  // below
  // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE
  if (voxel_dis_method == 1) {
    // Fuse
    constexpr float measurement_weight = 1.0f;
    float fused_distance = (voxel_distance_measured * measurement_weight +
                            voxel_distance_current * voxel_weight_current) /
                           (measurement_weight + voxel_weight_current);
    // Clip
    if (fused_distance > 0.0f) {
      fused_distance = fminf(truncation_distance_m, fused_distance);
    } else {
      fused_distance = fmaxf(-truncation_distance_m, fused_distance);
    }
    const float weight =
        fminf(measurement_weight + voxel_weight_current, max_weight);
    // Write NEW voxel values (to global GPU memory)
    voxel_ptr->distance = fused_distance;
    voxel_ptr->weight = weight;
  } else if (voxel_dis_method == 2) {
    voxel_distance_measured =
        fminf(voxel_distance_measured, truncation_distance_m);
    float measurement_weight = tsdf_constant_weight(voxel_distance_measured);
    float fused_distance = (voxel_distance_measured * measurement_weight +
                            voxel_distance_current * voxel_weight_current) /
                           (measurement_weight + voxel_weight_current);
    if (fused_distance > 0.0f) {
      fused_distance = fminf(truncation_distance_m, fused_distance);
    } else {
      fused_distance = fmaxf(-truncation_distance_m, fused_distance);
    }
    const float fused_weight =
        fminf(measurement_weight + voxel_weight_current, max_weight);
    voxel_ptr->distance = fused_distance;
    voxel_ptr->weight = fused_weight;
  } else if (voxel_dis_method == 3) {
    voxel_distance_measured =
        fminf(voxel_distance_measured, truncation_distance_m);
    float measurement_weight =
        tsdf_linear_weight(voxel_distance_measured, truncation_distance_m);
    float fused_distance = (voxel_distance_measured * measurement_weight +
                            voxel_distance_current * voxel_weight_current) /
                           (measurement_weight + voxel_weight_current);
    if (fused_distance > 0.0f) {
      fused_distance = fminf(truncation_distance_m, fused_distance);
    } else {
      fused_distance = fmaxf(-truncation_distance_m, fused_distance);
    }
    const float weight =
        fminf(measurement_weight + voxel_weight_current, max_weight);
    voxel_ptr->distance = fused_distance;
    voxel_ptr->weight = weight;
  } else if (voxel_dis_method == 4) {
    voxel_distance_measured =
        fminf(voxel_distance_measured, truncation_distance_m);
    float measurement_weight =
        tsdf_exp_weight(voxel_distance_measured, truncation_distance_m);
    float fused_distance = (voxel_distance_measured * measurement_weight +
                            voxel_distance_current * voxel_weight_current) /
                           (measurement_weight + voxel_weight_current);
    if (fused_distance > 0.0f) {
      fused_distance = fminf(truncation_distance_m, fused_distance);
    } else {
      fused_distance = fmaxf(-truncation_distance_m, fused_distance);
    }
    const float weight =
        fminf(measurement_weight + voxel_weight_current, max_weight);
    voxel_ptr->distance = fused_distance;
    voxel_ptr->weight = weight;
  } else if (voxel_dis_method == 5) {
    float normal_ratio = 1.0f;

    // case 1: existing gradient, use the gradient to compute the ratio
    if (voxel_gradient_current.norm() > kFloatEpsilon) {
      Vector3f gradient_C;
      // transform the gradient into the camera coordinate system
      gradient_C = T_C_L.rotation() * voxel_gradient_current;

      // case 1.1: existing gradient, existing normal
      if (measurement_normal.norm() > kFloatEpsilon) {
        // alpha: the angle between the normal and gradient
        float cos_alpha = abs(measurement_normal.dot(gradient_C));
        float sin_alpha = sqrt(1 - cos_alpha * cos_alpha);

        // theta: the angle between the ray and gradient
        float cos_theta =
            abs(measurement_point.dot(gradient_C) / measurement_point.norm());
        float sin_theta = sqrt(1 - cos_theta * cos_theta);

        // condition 1: flat surface, alpha is approximate to zero
        if (abs(1.0f - cos_alpha) < kFloatEpsilon) {
          normal_ratio = cos_theta;
        }
        // condition 2: curve surface
        else {
          normal_ratio =
              abs((cos_alpha - 1) * sin_theta / sin_alpha + cos_theta);
          if (isnan(normal_ratio)) normal_ratio = cos_theta;
        }
      }
      // case 1.2: existing gradient, no normal
      else {
        normal_ratio =
            abs(measurement_point.dot(gradient_C) / measurement_point.norm());
      }
    }
    // case 2:  no gradient
    else {
      // case 2.1: no gradient, existing normal
      if (measurement_normal.norm() > kFloatEpsilon) {
        normal_ratio = abs(measurement_point.dot(measurement_normal) /
                           measurement_point.norm());
      }
    }

    // ruling out extremely large incidence angle
    if (normal_ratio < 0.05) return false;

    float weight_sensor = tsdf_sensor_weight(surface_depth_measured, 2, 30.0);
    float weight_dropoff =
        tsdf_dropoff_weight(voxel_distance_measured, truncation_distance_m);
    float measurement_weight = weight_sensor * weight_dropoff;

    // NOTE(jjiao): it is possible to have weights very close to zero, due to
    // the limited precision of floating points dividing by this small value can
    // cause nans
    if (measurement_weight < kFloatEpsilon) return false;

    float measurement_distance = normal_ratio * voxel_distance_current;
    measurement_distance = fminf(measurement_distance, truncation_distance_m);
    float fused_distance = (measurement_distance * measurement_weight +
                            voxel_distance_current * voxel_weight_current) /
                           (measurement_weight + voxel_weight_current);
    const float fused_weight =
        fminf(voxel_weight_current + measurement_weight, max_weight);
    voxel_ptr->distance = fused_distance;
    voxel_ptr->weight = fused_weight;

    // existing normal, update the gradient
    if (measurement_normal.norm() > kFloatEpsilon) {
      Vector3f fused_gradient = (voxel_weight_current * voxel_gradient_current +
                                 measurement_weight * measurement_normal) /
                                (measurement_weight + voxel_weight_current);
      fused_gradient.normalize();
      voxel_ptr->gradient = fused_gradient;
    }
  }
  return true;
}

__device__ inline bool interpolateLidarImage(
    const Lidar& lidar, const Vector3f& p_voxel_center_C, const float* image,
    const Vector2f& u_px, const int rows, const int cols,
    const float linear_interpolation_max_allowable_difference_m,
    const float nearest_interpolation_max_allowable_squared_dist_to_ray_m,
    float* image_value) {
  // Try linear interpolation first
  interpolation::Interpolation2DNeighbours<float> neighbours;
  bool linear_interpolation_success = interpolation::interpolate2DLinear<
      float, interpolation::checkers::FloatPixelGreaterThanZero>(
      image, u_px, rows, cols, image_value, &neighbours);

  // Additional check
  // Check that we're not interpolating over a discontinuity
  // NOTE(alexmillane): This prevents smearing are object edges.
  if (linear_interpolation_success) {
    const float d00 = fabsf(neighbours.p00 - *image_value);
    const float d01 = fabsf(neighbours.p01 - *image_value);
    const float d10 = fabsf(neighbours.p10 - *image_value);
    const float d11 = fabsf(neighbours.p11 - *image_value);
    float maximum_depth_difference_to_neighbours =
        fmax(fmax(d00, d01), fmax(d10, d11));
    if (maximum_depth_difference_to_neighbours >
        linear_interpolation_max_allowable_difference_m) {
      linear_interpolation_success = false;
    }
  }

  // If linear didn't work - try nearest neighbour interpolation
  if (!linear_interpolation_success) {
    Index2D u_neighbour_px;
    if (!interpolation::interpolate2DClosest<
            float, interpolation::checkers::FloatPixelGreaterThanZero>(
            image, u_px, rows, cols, image_value, &u_neighbour_px)) {
      // If we can't successfully do closest, fail to intgrate this voxel.
      return false;
    }
    // Additional check
    // Check that this voxel is close to the ray passing through the pixel.
    // Note(alexmillane): This is to prevent large numbers of voxels
    // being integrated by a single pixel at long ranges.
    const Vector3f closest_ray = lidar.vectorFromPixelIndices(u_neighbour_px);
    const float off_ray_squared_distance =
        (p_voxel_center_C - p_voxel_center_C.dot(closest_ray) * closest_ray)
            .squaredNorm();
    if (off_ray_squared_distance >
        nearest_interpolation_max_allowable_squared_dist_to_ray_m) {
      return false;
    }
  }

  // TODO(alexmillane): We should add clearing rays, even in the case both
  // interpolations fail.

  return true;
}

// nearest_interpolation_max_allowable_squared_dist_to_ray_m, default: 0.125**2
__device__ inline bool interpolateOSLidarImage(
    const OSLidar& lidar, const Vector3f& p_voxel_center_C, const float* image,
    const Vector2f& u_px, const int rows, const int cols,
    const float linear_interpolation_max_allowable_difference_m,
    const float nearest_interpolation_max_allowable_squared_dist_to_ray_m,
    float* image_value) {
  // Try linear interpolation first
  interpolation::Interpolation2DNeighbours<float> neighbours;
  bool linear_interpolation_success = interpolation::interpolate2DLinear<
      float, interpolation::checkers::FloatPixelGreaterThanZero>(
      image, u_px, rows, cols, image_value, &neighbours);

  // Additional check
  // Check that we're not interpolating over a discontinuity
  // NOTE(alexmillane): This prevents smearing are object edges.
  if (linear_interpolation_success) {
    const float d00 = fabsf(neighbours.p00 - *image_value);
    const float d01 = fabsf(neighbours.p01 - *image_value);
    const float d10 = fabsf(neighbours.p10 - *image_value);
    const float d11 = fabsf(neighbours.p11 - *image_value);
    float maximum_depth_difference_to_neighbours =
        fmax(fmax(d00, d01), fmax(d10, d11));
    if (maximum_depth_difference_to_neighbours >
        linear_interpolation_max_allowable_difference_m) {
      linear_interpolation_success = false;
    }
  }

  // If linear didn't work - try nearest neighbour interpolation
  if (!linear_interpolation_success) {
    Index2D u_neighbour_px;
    if (!interpolation::interpolate2DClosest<
            float, interpolation::checkers::FloatPixelGreaterThanZero>(
            image, u_px, rows, cols, image_value, &u_neighbour_px)) {
      // If we can't successfully do closest, fail to intgrate this voxel.
      return false;
    }

    // Additional check
    // Check that this voxel is close to the ray passing through the pixel.
    // Note(alexmillane): This is to prevent large numbers of voxels
    // being integrated by a single pixel at long ranges.
    const Vector3f closest_ray = lidar.vectorFromPixelIndices(u_neighbour_px);
    const float off_ray_squared_distance =
        (p_voxel_center_C - p_voxel_center_C.dot(closest_ray) * closest_ray)
            .squaredNorm();
    if (off_ray_squared_distance >
        nearest_interpolation_max_allowable_squared_dist_to_ray_m) {
      return false;
    }
  }

  // TODO(alexmillane): We should add clearing rays, even in the case both
  // interpolations fail.
  return true;
}

// NOTE(jjiao):
__device__ inline bool getPointVectorOSLidar(const OSLidar& lidar,
                                             const Index2D& u_C, const int rows,
                                             const int cols,
                                             Vector3f& point_vector) {
  if (u_C.x() < 0 || u_C.y() < 0 || u_C.x() >= cols || u_C.y() >= rows) {
    return false;
  } else {
    point_vector = lidar.unprojectFromImageIndex(u_C);
    if (point_vector.norm() < kFloatEpsilon) {
      return false;
    } else {
      return true;
    }
  }
}

// NOTE(jjiao):
__device__ inline bool getNormalVectorOSLidar(const OSLidar& lidar,
                                              const Index2D& u_C,
                                              const int rows, const int cols,
                                              Vector3f& normal_vector) {
  if (u_C.x() < 0 || u_C.y() < 0 || u_C.x() >= cols || u_C.y() >= rows) {
    return false;
  } else {
    normal_vector = lidar.getNormalVector(u_C);
    if (normal_vector.norm() < kFloatEpsilon) {
      return false;
    } else {
      return true;
    }
  }
}

// CAMERA
__global__ void integrateBlocksKernel(const Index3D* block_indices_device_ptr,
                                      const Camera camera, const float* image,
                                      int rows, int cols, const Transform T_C_L,
                                      const float block_size,
                                      const float truncation_distance_m,
                                      const float max_weight,
                                      const float max_integration_distance,
                                      TsdfBlock** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this
  // thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel(block_indices_device_ptr, camera, T_C_L, block_size,
                          &u_px, &voxel_depth_m, &p_voxel_center_C)) {
    return;
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  // Interpolate on the image plane
  float image_value;
  if (!interpolation::interpolate2DClosest<
          float, interpolation::checkers::FloatPixelGreaterThanZero>(
          image, u_px, rows, cols, &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order
  // such that adjacent threads (x-major) access adjacent memory locations
  // in the block (z-major).
  TsdfVoxel* voxel_ptr = &(block_device_ptrs[blockIdx.x]
                               ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type
  updateVoxel(image_value, voxel_ptr, voxel_depth_m, truncation_distance_m,
              max_weight);
}

// LIDAR
__global__ void integrateBlocksKernel(
    const Index3D* block_indices_device_ptr, const Lidar lidar,
    const float* image, int rows, int cols, const Transform T_C_L,
    const float block_size, const float truncation_distance_m,
    const float max_weight, const float max_integration_distance,
    const float linear_interpolation_max_allowable_difference_m,
    const float nearest_interpolation_max_allowable_squared_dist_to_ray_m,
    TsdfBlock** block_device_ptrs) {
  // Get - the image-space projection of the voxel associated with this
  // thread
  //     - the depth associated with the projection.
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel(block_indices_device_ptr, lidar, T_C_L, block_size,
                          &u_px, &voxel_depth_m, &p_voxel_center_C)) {
    return;
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  // Interpolate on the image plane
  float image_value;
  if (!interpolateLidarImage(
          lidar, p_voxel_center_C, image, u_px, rows, cols,
          linear_interpolation_max_allowable_difference_m,
          nearest_interpolation_max_allowable_squared_dist_to_ray_m,
          &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order
  // such that adjacent threads (x-major) access adjacent memory locations
  // in the block (z-major).
  TsdfVoxel* voxel_ptr = &(block_device_ptrs[blockIdx.x]
                               ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // Update the voxel using the update rule for this layer type
  updateVoxel(image_value, voxel_ptr, voxel_depth_m, truncation_distance_m,
              max_weight);
}

// OSLiDAR
// TODO(jjiao): main function to integrate blocks in GPU
__global__ void integrateBlocksKernel(
    const Index3D* block_indices_device_ptr, const OSLidar lidar,
    const float* image, int rows, int cols, const Transform T_C_L,
    const float block_size, const float truncation_distance_m,
    const float max_weight, const float max_integration_distance,
    const float linear_interpolation_max_allowable_difference_m,
    const float nearest_interpolation_max_allowable_squared_dist_to_ray_m,
    TsdfBlock** block_device_ptrs) {
  // function 1
  // Get - the image-space projection of the voxel associated with this
  // thread
  //     - the depth associated with the projection.
  //     - the projected image coordinate of the voxel
  Eigen::Vector2f u_px;
  float voxel_depth_m;
  Vector3f p_voxel_center_C;
  if (!projectThreadVoxel(block_indices_device_ptr, lidar, T_C_L, block_size,
                          &u_px, &voxel_depth_m, &p_voxel_center_C)) {
    // TODO(jjiao): please remove these sentences after debugging
    // printf("u(%.2f, %.2f), p(%.2f, %.2f, %.2f), dep(%.2f)\n", u_px.x(),
    //        u_px.y(), p_voxel_center_C.x(), p_voxel_center_C.y(),
    //        p_voxel_center_C.z(), voxel_depth_m);
    return;  // false: the voxel is not visible
  }

  // If voxel further away than the limit, skip this voxel
  if (max_integration_distance > 0.0f) {
    if (voxel_depth_m > max_integration_distance) {
      return;
    }
  }

  // function 2
  // Interpolate on the image plane
  float image_value;
  if (!interpolateOSLidarImage(
          lidar, p_voxel_center_C, image, u_px, rows, cols,
          linear_interpolation_max_allowable_difference_m,
          nearest_interpolation_max_allowable_squared_dist_to_ray_m,
          &image_value)) {
    return;
  }

  // Get the Voxel we'll update in this thread
  // NOTE(alexmillane): Note that we've reverse the voxel indexing order
  // such that adjacent threads (x-major) access adjacent memory locations
  // in the block (z-major).
  TsdfVoxel* voxel_ptr = &(block_device_ptrs[blockIdx.x]
                               ->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);

  // NOTE(jjiao): retrive the normal vector given u_px
  const Index2D u_C = u_px.array().round().cast<int>();
  Vector3f point_vector = Vector3f::Zero();
  Vector3f normal_vector = Vector3f::Zero();
  if (!getPointVectorOSLidar(lidar, u_C, rows, cols, point_vector)) return;
  if (!getNormalVectorOSLidar(lidar, u_C, rows, cols, normal_vector)) return;
  normal_vector = T_C_L.rotation().transpose() * normal_vector;

  // function 3
  // Update the voxel using the update rule for this layer type
  // NOTE(jjiao):
  // setting the voxel update method
  // Projective distance:
  //  1: constant weight, truncate the fused_distance
  //  2: constant weight, truncate the voxel_distance_measured
  //  3: linear weight, truncate the voxel_distance_measured
  //  4: exponential weight, truncate the voxel_distance_measured
  // Non-Projective distance:
  //  5: weight and distance derived from VoxField
  const int voxel_dis_method = 3;
  if (voxel_dis_method == 1) {
    // the original nvblox impelentation
    // not use normal vector
    updateVoxel(image_value, voxel_ptr, voxel_depth_m, truncation_distance_m,
                max_weight);
  } else {
    // the improved weight computation
    // use normal vector
    updateVoxelMultiWeightComp(
        image_value, voxel_ptr, voxel_depth_m, truncation_distance_m,
        max_weight, voxel_dis_method, point_vector, normal_vector, T_C_L);
  }
}

ProjectiveTsdfIntegrator::ProjectiveTsdfIntegrator()
    : ProjectiveIntegratorBase() {
  checkCudaErrors(cudaStreamCreate(&integration_stream_));
}

ProjectiveTsdfIntegrator::~ProjectiveTsdfIntegrator() {
  finish();
  checkCudaErrors(cudaStreamDestroy(integration_stream_));
}

void ProjectiveTsdfIntegrator::finish() const {
  cudaStreamSynchronize(integration_stream_);
}

float ProjectiveTsdfIntegrator::
    lidar_linear_interpolation_max_allowable_difference_vox() const {
  return lidar_linear_interpolation_max_allowable_difference_vox_;
}

float ProjectiveTsdfIntegrator::
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox() const {
  return lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_;
}

void ProjectiveTsdfIntegrator::
    lidar_linear_interpolation_max_allowable_difference_vox(float value) {
  CHECK_GT(value, 0.0f);
  lidar_linear_interpolation_max_allowable_difference_vox_ = value;
}

void ProjectiveTsdfIntegrator::
    lidar_nearest_interpolation_max_allowable_dist_to_ray_vox(float value) {
  CHECK_GT(value, 0.0f);
  lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ = value;
}

template <typename SensorType>
void ProjectiveTsdfIntegrator::integrateFrameTemplate(
    const DepthImage& depth_frame, const Transform& T_L_C,
    const SensorType& sensor, TsdfLayer* layer,
    std::vector<Index3D>* updated_blocks) {
  CHECK_NOTNULL(layer);
  timing::Timer tsdf_timer("tsdf/integrate");

  // Metric truncation distance for this layer
  const float voxel_size =
      layer->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  // Identify blocks we can (potentially) see
  timing::Timer blocks_in_view_timer("tsdf/integrate/get_blocks_in_view");
  const std::vector<Index3D> block_indices =
      view_calculator_.getBlocksInImageViewRaycast(
          depth_frame, T_L_C, sensor, layer->block_size(),
          truncation_distance_m, max_integration_distance_m_);
  // LOG(INFO) << "block_indices size: " << block_indices.size();
  blocks_in_view_timer.Stop();

  // Allocate blocks (CPU)
  timing::Timer allocate_blocks_timer("tsdf/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, layer);
  allocate_blocks_timer.Stop();

  // Update identified blocks
  timing::Timer update_blocks_timer("tsdf/integrate/update_blocks");
  integrateBlocksTemplate<SensorType>(block_indices, depth_frame, T_L_C, sensor,
                                      layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

// Camera
void ProjectiveTsdfIntegrator::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    TsdfLayer* layer, std::vector<Index3D>* updated_blocks) {
  integrateFrameTemplate(depth_frame, T_L_C, camera, layer, updated_blocks);
}

// Lidar
void ProjectiveTsdfIntegrator::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Lidar& lidar,
    TsdfLayer* layer, std::vector<Index3D>* updated_blocks) {
  integrateFrameTemplate(depth_frame, T_L_C, lidar, layer, updated_blocks);
}

// OSLidar
void ProjectiveTsdfIntegrator::integrateFrame(
    DepthImage& depth_frame, const Transform& T_L_C, OSLidar& oslidar,
    TsdfLayer* layer, std::vector<Index3D>* updated_blocks) {
  integrateFrameTemplate(depth_frame, T_L_C, oslidar, layer, updated_blocks);
}

// Camera
void ProjectiveTsdfIntegrator::integrateBlocks(const DepthImage& depth_frame,
                                               const Transform& T_C_L,
                                               const Camera& camera,
                                               TsdfLayer* layer_ptr) {
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = block_indices_device_.size();

  // Metric truncation distance for this layer
  const float voxel_size =
      layer_ptr->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  // Kernel
  integrateBlocksKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                          integration_stream_>>>(
      block_indices_device_.data(),  // NOLINT
      camera,                        // NOLINT
      depth_frame.dataConstPtr(),    // NOLINT
      depth_frame.rows(),            // NOLINT
      depth_frame.cols(),            // NOLINT
      T_C_L,                         // NOLINT
      layer_ptr->block_size(),       // NOLINT
      truncation_distance_m,         // NOLINT
      max_weight_,                   // NOLINT
      max_integration_distance_m_,   // NOLINT
      block_ptrs_device_.data());    // NOLINT

  // Finish processing of the frame before returning control
  finish();
  checkCudaErrors(cudaPeekAtLastError());
}

// Lidar
void ProjectiveTsdfIntegrator::integrateBlocks(const DepthImage& depth_frame,
                                               const Transform& T_C_L,
                                               const Lidar& lidar,
                                               TsdfLayer* layer_ptr) {
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = block_indices_device_.size();

  // Metric truncation distance for this layer
  const float voxel_size =
      layer_ptr->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  // Metric params
  const float linear_interpolation_max_allowable_difference_m =
      lidar_linear_interpolation_max_allowable_difference_vox_ * voxel_size;
  const float nearest_interpolation_max_allowable_squared_dist_to_ray_m =
      std::pow(lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ *
                   voxel_size,
               2);

  // Kernel
  integrateBlocksKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                          integration_stream_>>>(
      block_indices_device_.data(),                               // NOLINT
      lidar,                                                      // NOLINT
      depth_frame.dataConstPtr(),                                 // NOLINT
      depth_frame.rows(),                                         // NOLINT
      depth_frame.cols(),                                         // NOLINT
      T_C_L,                                                      // NOLINT
      layer_ptr->block_size(),                                    // NOLINT
      truncation_distance_m,                                      // NOLINT
      max_weight_,                                                // NOLINT
      max_integration_distance_m_,                                // NOLINT
      linear_interpolation_max_allowable_difference_m,            // NOLINT
      nearest_interpolation_max_allowable_squared_dist_to_ray_m,  // NOLINT
      block_ptrs_device_.data());                                 // NOLINT

  // Finish processing of the frame before returning control
  finish();
  checkCudaErrors(cudaPeekAtLastError());
}

// OSLidar
void ProjectiveTsdfIntegrator::integrateBlocks(const DepthImage& depth_frame,
                                               const Transform& T_C_L,
                                               const OSLidar& lidar,
                                               TsdfLayer* layer_ptr) {
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  // NOTE(jjiao): the number of visible blocks
  const int num_thread_blocks = block_indices_device_.size();

  // Metric truncation distance for this layer
  const float voxel_size =
      layer_ptr->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m = truncation_distance_vox_ * voxel_size;

  // Metric params
  const float linear_interpolation_max_allowable_difference_m =
      lidar_linear_interpolation_max_allowable_difference_vox_ * voxel_size;
  const float nearest_interpolation_max_allowable_squared_dist_to_ray_m =
      std::pow(lidar_nearest_interpolation_max_allowable_dist_to_ray_vox_ *
                   voxel_size,
               2);

  // Kernel
  // std::cout << "num_thread_blocks: " << num_thread_blocks << std::endl;
  // std::cout << "kVoxelsPerSide: " << kVoxelsPerSide << std::endl;
  integrateBlocksKernel<<<num_thread_blocks, kThreadsPerBlock, 0,
                          integration_stream_>>>(
      block_indices_device_.data(),                               // NOLINT
      lidar,                                                      // NOLINT
      depth_frame.dataConstPtr(),                                 // NOLINT
      depth_frame.rows(),                                         // NOLINT
      depth_frame.cols(),                                         // NOLINT
      T_C_L,                                                      // NOLINT
      layer_ptr->block_size(),                                    // NOLINT
      truncation_distance_m,                                      // NOLINT
      max_weight_,                                                // NOLINT
      max_integration_distance_m_,                                // NOLINT
      linear_interpolation_max_allowable_difference_m,            // NOLINT
      nearest_interpolation_max_allowable_squared_dist_to_ray_m,  // NOLINT
      block_ptrs_device_.data());                                 // NOLINT

  // Finish processing of the frame before returning control
  finish();
  checkCudaErrors(cudaPeekAtLastError());
}

template <typename SensorType>
void ProjectiveTsdfIntegrator::integrateBlocksTemplate(
    const std::vector<Index3D>& block_indices, const DepthImage& depth_frame,
    const Transform& T_L_C, const SensorType& sensor, TsdfLayer* layer_ptr) {
  CHECK_NOTNULL(layer_ptr);

  if (block_indices.empty()) {
    return;
  }
  const int num_blocks = block_indices.size();

  // Expand the buffers when needed
  if (num_blocks > block_indices_device_.size()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    block_indices_device_.reserve(new_size);
    block_ptrs_device_.reserve(new_size);
    block_indices_host_.reserve(new_size);
    block_ptrs_host_.reserve(new_size);
  }

  // Stage on the host pinned memory
  block_indices_host_ = block_indices;
  block_ptrs_host_ = getBlockPtrsFromIndices(block_indices, layer_ptr);

  // Transfer to the device
  block_indices_device_ = block_indices_host_;
  block_ptrs_device_ = block_ptrs_host_;

  // We need the inverse transform in the kernel
  const Transform T_C_L = T_L_C.inverse();

  // Calling the GPU to do the updates
  integrateBlocks(depth_frame, T_C_L, sensor, layer_ptr);
}

}  // namespace nvblox
