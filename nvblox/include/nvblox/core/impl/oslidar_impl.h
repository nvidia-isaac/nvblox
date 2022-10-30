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
NOTE(jjiao):
This implements the operation that project OSLidar points onto a depth image
*/

#pragma once

#include "math.h"

#include <glog/logging.h>

namespace nvblox {

// default: 2048, 128, 45
OSLidar::OSLidar(int num_azimuth_divisions, int num_elevation_divisions,
                 float vertical_fov_rad, float horizontal_fov_rad,
                 std::shared_ptr<CoordImage>& coord_image_ptr)
    : num_azimuth_divisions_(num_azimuth_divisions),
      num_elevation_divisions_(num_elevation_divisions),
      vertical_fov_rad_(vertical_fov_rad),
      horizontal_fov_rad_(horizontal_fov_rad) {
  // Even numbers of beams allowed
  CHECK(num_azimuth_divisions_ % 2 == 0);

  // Angular distance between pixels
  // Note(alexmillane): Note the difference in division by N vs. (N-1) below.
  // This is because in the azimuth direction there's a wrapping around. The
  // point at pi/-pi is not double sampled, generating this difference.
  rads_per_pixel_elevation_ =
      vertical_fov_rad / static_cast<float>(num_elevation_divisions_ - 1);
  rads_per_pixel_azimuth_ =
      horizontal_fov_rad_ / static_cast<float>(num_azimuth_divisions_ - 1);

  // Inverse of the above
  elevation_pixels_per_rad_ = 1.0f / rads_per_pixel_elevation_;
  azimuth_pixels_per_rad_ = 1.0f / rads_per_pixel_azimuth_;

  // The angular lower-extremes of the image-plane
  // NOTE(alexmillane): Because beams pass through the angular extremes of the
  // FoV, the corresponding lower extreme pixels start half a pixel width
  // below this.
  // Note(alexmillane): Note that we use polar angle here, not elevation.
  // Polar is from the top of the sphere down, elevation, the middle up.
  // start_polar_angle_rad_ = M_PI / 2.0f - (vertical_fov_rad / 2.0f +
  //                                         rads_per_pixel_elevation_ / 2.0f);
  // start_azimuth_angle_rad_ = -M_PI - rads_per_pixel_azimuth_ / 2.0f;

  // NOTE(jjiao): define the depth image
  // the start azimuth_angle indicate the direction: x=0, +y
  // the end azimuth_angle indicate the direction: x=0, -y
  start_polar_angle_rad_ = vertical_fov_rad / 2.0f;
  start_azimuth_angle_rad_ = 0;

  coord_image_ptr_ = coord_image_ptr;
}

int OSLidar::num_azimuth_divisions() const { return num_azimuth_divisions_; }

int OSLidar::num_elevation_divisions() const {
  return num_elevation_divisions_;
}

float OSLidar::vertical_fov_rad() const { return vertical_fov_rad_; }

int OSLidar::numel() const {
  return num_azimuth_divisions_ * num_elevation_divisions_;
}

int OSLidar::cols() const { return num_azimuth_divisions_; }

int OSLidar::rows() const { return num_elevation_divisions_; }

bool OSLidar::project(const Vector3f& p_C, Vector2f* u_C) const {
  const float rxy = p_C.head<2>().norm();
  constexpr float kMinProjectionEps = 0.01;
  if (p_C.norm() < kMinProjectionEps) {
    return false;
  }
  const float polar_angle_rad = atan2(p_C.z(), rxy);
  const float azimuth_angle_rad = M_PI - atan2(p_C.y(), p_C.x());

  // To image plane coordinates
  // very close to the row_id and column_id (int)
  float v_float =
      (start_polar_angle_rad_ - polar_angle_rad) / rads_per_pixel_elevation_;
  float u_float =
      (azimuth_angle_rad - start_azimuth_angle_rad_) / rads_per_pixel_azimuth_;

  // float v_float =
  //     (polar_angle_rad - start_polar_angle_rad_) * elevation_pixels_per_rad_;
  // float u_float =
  //     (azimuth_angle_rad - start_azimuth_angle_rad_) *
  //     azimuth_pixels_per_rad_;

  // Catch wrap around issues.
  if (u_float >= num_azimuth_divisions_) {
    u_float -= num_azimuth_divisions_;
  }

  // Points out of FOV
  // NOTE(alexmillane): It should be impossible to escape the -pi-to-pi range in
  // azimuth due to wrap around this. Therefore we don't check.
  if (v_float < -kMinProjectionEps ||
      v_float >= num_elevation_divisions_ + kMinProjectionEps) {
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

float OSLidar::getDepth(const Vector3f& p_C) const { return p_C.norm(); }

Vector2f OSLidar::pixelIndexToImagePlaneCoordsOfCenter(
    const Index2D& u_C) const {
  // The index cast to a float is the coordinates of the lower corner of the
  // pixel.
  return u_C.cast<float>() + Vector2f(0.5f, 0.5f);
}

Index2D OSLidar::imagePlaneCoordsToPixelIndex(const Vector2f& u_C) const {
  // NOTE(alexmillane): We do round rather than a straight truncation such that
  // we handle negative image plane coordinates.
  return u_C.array().round().cast<int>();
}

Vector3f OSLidar::unprojectFromImagePlaneCoordinates(const Vector2f& u_C,
                                                     const float depth) const {
  return vectorFromImagePlaneCoordinates(u_C);
}

Vector3f OSLidar::unprojectFromPixelIndices(const Index2D& u_C,
                                            const float depth) const {
  return vectorFromPixelIndices(u_C);
}

Vector3f OSLidar::vectorFromImagePlaneCoordinates(const Vector2f& u_C) const {
  return (*coord_image_ptr_)(round(u_C.y()), round(u_C.x()));
}

Vector3f OSLidar::vectorFromPixelIndices(const Index2D& u_C) const {
  return (*coord_image_ptr_)(u_C.y(), u_C.x());
}

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
