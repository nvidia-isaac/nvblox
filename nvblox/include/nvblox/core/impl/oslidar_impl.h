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

/*
NOTE(gogojjh):
This implements the operation that project OSLidar points onto a depth image
*/

#pragma once

#include "math.h"

#include <glog/logging.h>

namespace nvblox {

// given initial intrinsics from the intrisics file, but need to be refined for
// each new scan
OSLidar::OSLidar(int num_azimuth_divisions, int num_elevation_divisions,
                 float horizontal_fov_rad, float vertical_fov_rad,
                 float start_azimuth_angle_rad, float end_azimuth_angle_rad,
                 float start_elevation_angle_rad, float end_elevation_angle_rad)
    : num_azimuth_divisions_(num_azimuth_divisions),
      num_elevation_divisions_(num_elevation_divisions),
      horizontal_fov_rad_(horizontal_fov_rad),
      vertical_fov_rad_(vertical_fov_rad),
      start_azimuth_angle_rad_(start_azimuth_angle_rad),
      end_azimuth_angle_rad_(end_azimuth_angle_rad),
      start_elevation_angle_rad_(start_elevation_angle_rad),
      end_elevation_angle_rad_(end_elevation_angle_rad) {
  // Even numbers of beams allowed
  CHECK(num_azimuth_divisions_ % 2 == 0);

  rads_per_pixel_elevation_ =
      vertical_fov_rad_ / static_cast<float>(num_elevation_divisions_ - 1);
  rads_per_pixel_azimuth_ =
      horizontal_fov_rad_ / static_cast<float>(num_azimuth_divisions_ - 1);

  // Inverse of the above
  elevation_pixels_per_rad_ = 1.0f / rads_per_pixel_elevation_;
  azimuth_pixels_per_rad_ = 1.0f / rads_per_pixel_azimuth_;

  // printIntrinsics();

  depth_image_ptr_cuda_ = nullptr;
  height_image_ptr_cuda_ = nullptr;
  normal_image_ptr_cuda_ = nullptr;
}

OSLidar::~OSLidar() {}

void OSLidar::printIntrinsics() const {
  printf("OSLidar intrinsics--------------------\n");
  printf("horizontal_fov_rad: %f\n", horizontal_fov_rad_);
  printf("vertical_fov_rad: %f\n", vertical_fov_rad_);
  printf("start_elevation: %f\n", start_elevation_angle_rad_);
  printf("end_elevation: %f\n", end_elevation_angle_rad_);
  printf("rads_per_pixel_elevation: %f\n", rads_per_pixel_elevation_);
  printf("rads_per_pixel_azimuth: %f\n", rads_per_pixel_azimuth_);
}

/**********************************************
 * get the parameters of OSLidar
 **********************************************/
int OSLidar::num_azimuth_divisions() const { return num_azimuth_divisions_; }

int OSLidar::num_elevation_divisions() const {
  return num_elevation_divisions_;
}

float OSLidar::vertical_fov_rad() const { return vertical_fov_rad_; }

float OSLidar::horizontal_fov_rad() const { return horizontal_fov_rad_; }

float OSLidar::rads_per_pixel_elevation() const {
  return rads_per_pixel_elevation_;
}

float OSLidar::rads_per_pixel_azimuth() const {
  return rads_per_pixel_azimuth_;
}

int OSLidar::numel() const {
  return num_azimuth_divisions_ * num_elevation_divisions_;
}

int OSLidar::cols() const { return num_azimuth_divisions_; }

int OSLidar::rows() const { return num_elevation_divisions_; }

float OSLidar::getDepth(const Vector3f& p_C) const { return p_C.norm(); }

// NOTE(gogojjh): this function is added by gogojjh
Vector3f OSLidar::getNormalVector(const Index2D& u_C) const {
  if (normal_image_ptr_cuda_) {
    float x = normal_image_ptr_cuda_[3 * (u_C.y() * num_azimuth_divisions_ +
                                          u_C.x())];
    float y = normal_image_ptr_cuda_[3 * (u_C.y() * num_azimuth_divisions_ +
                                          u_C.x()) +
                                     1];
    float z = normal_image_ptr_cuda_[3 * (u_C.y() * num_azimuth_divisions_ +
                                          u_C.x()) +
                                     2];
    return Vector3f(x, y, z);
  } else {
    return Vector3f(0.0f, 0.0f, 0.0f);
  }
}

/**********************************************
 * Project a 3D point p_C to get the image coordinates u_C
 **********************************************/
bool OSLidar::project(const Vector3f& p_C, Vector2f* u_C) const {
  const float r = p_C.norm();
  constexpr float kMinProjectionEps = 0.01;
  if (r < kMinProjectionEps) return false;

  const float elevation_angle_rad = acos(p_C.z() / r);
  const float azimuth_angle_rad = M_PI - atan2(p_C.y(), p_C.x());
  float v_float = (elevation_angle_rad - start_elevation_angle_rad_) /
                  rads_per_pixel_elevation_;
  float u_float =
      (azimuth_angle_rad - start_azimuth_angle_rad_) / rads_per_pixel_azimuth_;

  // Catch wrap around issues.
  if (u_float >= num_azimuth_divisions_) {
    u_float -= num_azimuth_divisions_;
  }

  // Points out of FOV
  // NOTE(alexmillane): It should be impossible to escape the -pi-to-pi range in
  // azimuth due to wrap around this. Therefore we don't check.
  if ((round(v_float) < 0) || (round(v_float) > num_elevation_divisions_ - 1)) {
    return false;
  }

  // Write output
  *u_C = Vector2f(u_float, v_float);
  return true;
}

bool OSLidar::project(const Vector3f& p_C, Index2D* u_C) const {
  Vector2f u_C_float;
  bool res = project(p_C, &u_C_float);
  *u_C = u_C_float.array().round().matrix().cast<int>();
  return res;
}

/**********************************************
 * transformation between the pixel index (int) and image coordinates (float)
 **********************************************/
Vector2f OSLidar::pixelIndexToImagePlaneCoordsOfCenter(
    const Index2D& u_C) const {
  // The index cast to a float is the coordinates of the lower corner of the
  // pixel.
  return u_C.cast<float>();
}

Index2D OSLidar::imagePlaneCoordsToPixelIndex(const Vector2f& u_C) const {
  // NOTE(alexmillane): We do round rather than a straight truncation such that
  // we handle negative image plane coordinates.
  return u_C.array().round().cast<int>();
}

/**********************************************
 * Unproject a 2D image coordinates u_C to a 3D point p_C
 **********************************************/
Vector3f OSLidar::unprojectFromImagePlaneCoordinates(const Vector2f& u_C,
                                                     const float depth) const {
  return depth * vectorFromImagePlaneCoordinates(u_C);
}

Vector3f OSLidar::unprojectFromPixelIndices(const Index2D& u_C,
                                            const float depth) const {
  return depth * vectorFromPixelIndices(u_C);
}

Vector3f OSLidar::unprojectFromImageIndex(const Index2D& u_C) const {
  float height = image::access<float>(u_C.y(), u_C.x(), num_azimuth_divisions_,
                                      height_image_ptr_cuda_);
  float depth = image::access<float>(u_C.y(), u_C.x(), num_azimuth_divisions_,
                                     depth_image_ptr_cuda_);
  float r = sqrt(depth * depth - height * height);
  float azimuth_angle_rad = M_PI - u_C.x() * rads_per_pixel_azimuth_;
  Vector3f p(r * cos(azimuth_angle_rad), r * sin(azimuth_angle_rad), height);
  return p;
}

Vector3f OSLidar::vectorFromImagePlaneCoordinates(const Vector2f& u_C) const {
  float height =
      image::access<float>(round(u_C.y()), round(u_C.x()),
                           num_azimuth_divisions_, height_image_ptr_cuda_);
  float depth =
      image::access<float>(round(u_C.y()), round(u_C.x()),
                           num_azimuth_divisions_, depth_image_ptr_cuda_);
  float r = sqrt(depth * depth - height * height);
  float azimuth_angle_rad = M_PI - u_C.x() * rads_per_pixel_azimuth_;
  Vector3f p(r * cos(azimuth_angle_rad), r * sin(azimuth_angle_rad), height);
  return p / depth;
}

Vector3f OSLidar::vectorFromPixelIndices(const Index2D& u_C) const {
  return vectorFromImagePlaneCoordinates(
      pixelIndexToImagePlaneCoordsOfCenter(u_C));
}

/**********************************************
 * Project a 3D point p_C to get the image coordinates u_C
 **********************************************/
AxisAlignedBoundingBox OSLidar::getViewAABB(const Transform& T_L_C,
                                            const float min_depth,
                                            const float max_depth) const {
  // The AABB is a square centered at the OSLidars location where the height is
  // determined by the OSLidar FoV.
  // NOTE(alexmillane): The min depth is ignored in this function, it is a
  // parameter so it matches with camera's getViewAABB()
  AxisAlignedBoundingBox box(
      Vector3f(-max_depth, -max_depth,
               -max_depth * sin(vertical_fov_rad_ / 2.0f)),
      Vector3f(max_depth, max_depth,
               max_depth * sin(vertical_fov_rad_ / 2.0f)));

  // Translate the box to the sensor's location (note that orientation doesn't
  // matter as the OSLidar sees in the circle)
  box.translate(T_L_C.translation());
  return box;
}

size_t OSLidar::Hash::operator()(const OSLidar& OSLidar) const {
  // Taken from:
  // https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
  size_t az_hash = std::hash<int>()(OSLidar.num_azimuth_divisions_);
  size_t el_hash = std::hash<int>()(OSLidar.num_elevation_divisions_);
  size_t fov_hash = std::hash<float>()(OSLidar.vertical_fov_rad_);
  return ((az_hash ^ (el_hash << 1)) >> 1) ^ (fov_hash << 1);
}

bool operator==(const OSLidar& lhs, const OSLidar& rhs) {
  return (lhs.num_azimuth_divisions_ == rhs.num_azimuth_divisions_) &&
         (lhs.num_elevation_divisions_ == rhs.num_elevation_divisions_) &&
         (std::fabs(lhs.vertical_fov_rad_ - rhs.vertical_fov_rad_) <
          std::numeric_limits<float>::epsilon());
}
}  // namespace nvblox
