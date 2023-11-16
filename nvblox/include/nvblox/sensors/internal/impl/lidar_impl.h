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

#include "math.h"

#include "nvblox/utils/logging.h"

namespace nvblox {

Lidar::Lidar(int num_azimuth_divisions, int num_elevation_divisions,
             float vertical_fov_rad)
    : Lidar(num_azimuth_divisions, num_elevation_divisions,
            vertical_fov_rad / 2.0f, vertical_fov_rad / 2.0f) {}

Lidar::Lidar(int num_azimuth_divisions, int num_elevation_divisions,
             float min_angle_below_zero_elevation_rad,
             float max_angle_above_zero_elevation_rad)
    : num_azimuth_divisions_(num_azimuth_divisions),
      num_elevation_divisions_(num_elevation_divisions) {
  // Only even numbers of azimuth divisions allowed
  CHECK(num_azimuth_divisions_ % 2 == 0);

  // Max sure the min elevation angle is negative (this also makes us tolerant
  // to positive and negative inputs).
  if (min_angle_below_zero_elevation_rad > 0) {
    min_angle_below_zero_elevation_rad = -min_angle_below_zero_elevation_rad;
  }
  CHECK_GT(max_angle_above_zero_elevation_rad, 0.f);
  CHECK_LT(min_angle_below_zero_elevation_rad, 0.f);

  // Calculate the vertical FOV
  vertical_fov_rad_ =
      max_angle_above_zero_elevation_rad - min_angle_below_zero_elevation_rad;

  // Angular distance between pixels
  // Note(alexmillane): Note the difference in division by N vs. (N-1) below.
  // This is because in the azimuth direction there's a wrapping around. The
  // point at pi/-pi is not double sampled, generating this difference.
  rads_per_pixel_elevation_ =
      vertical_fov_rad_ / static_cast<float>(num_elevation_divisions_ - 1);
  rads_per_pixel_azimuth_ =
      2.0f * M_PI / static_cast<float>(num_azimuth_divisions_);

  // Inverse of the above
  elevation_pixels_per_rad_ = 1.0f / rads_per_pixel_elevation_;
  azimuth_pixels_per_rad_ = 1.0f / rads_per_pixel_azimuth_;

  // The angular lower-extremes of the image-plane
  // NOTE(alexmillane): Because beams pass through the angular extremes of the
  // FoV, the corresponding lower extreme pixels start half a pixel width
  // below this.
  // Note(alexmillane): Note that we use polar angle here, not elevation.
  // Polar is from the top of the sphere down, elevation, the middle up.
  start_polar_angle_rad_ = M_PI / 2.0f - (max_angle_above_zero_elevation_rad +
                                          rads_per_pixel_elevation_ / 2.0f);
  start_azimuth_angle_rad_ = -M_PI - rads_per_pixel_azimuth_ / 2.0f;
}

int Lidar::num_azimuth_divisions() const { return num_azimuth_divisions_; }

int Lidar::num_elevation_divisions() const { return num_elevation_divisions_; }

float Lidar::vertical_fov_rad() const { return vertical_fov_rad_; }

float Lidar::start_polar_angle_rad() const { return start_polar_angle_rad_; }

int Lidar::numel() const {
  return num_azimuth_divisions_ * num_elevation_divisions_;
}

int Lidar::cols() const { return num_azimuth_divisions_; }

int Lidar::rows() const { return num_elevation_divisions_; }

bool Lidar::project(const Vector3f& p_C, Vector2f* u_C) const {
  // To spherical coordinates
  const float r = p_C.norm();
  constexpr float kMinProjectionEps = 0.01;
  if (r < kMinProjectionEps) {
    return false;
  }
  const float polar_angle_rad = acos(p_C.z() / r);
  const float azimuth_angle_rad = atan2(p_C.y(), p_C.x());

  // To image plane coordinates
  float v_float =
      (polar_angle_rad - start_polar_angle_rad_) * elevation_pixels_per_rad_;
  float u_float =
      (azimuth_angle_rad - start_azimuth_angle_rad_) * azimuth_pixels_per_rad_;

  // Catch wrap around issues.
  if (u_float >= num_azimuth_divisions_) {
    u_float -= num_azimuth_divisions_;
  }

  // Points out of FOV
  // NOTE(alexmillane): It should be impossible to escape the -pi-to-pi range in
  // azimuth due to wrap around this. Therefore we don't check.
  if (v_float < 0.0f || v_float >= num_elevation_divisions_) {
    return false;
  }

  // Write output
  *u_C = Vector2f(u_float, v_float);
  return true;
}

bool Lidar::project(const Vector3f& p_C, Index2D* u_C) const {
  Vector2f u_C_float;
  bool res = project(p_C, &u_C_float);
  *u_C = u_C_float.array().floor().matrix().cast<int>();
  return res;
}

float Lidar::getDepth(const Vector3f& p_C) const { return p_C.norm(); }

Vector2f Lidar::pixelIndexToImagePlaneCoordsOfCenter(const Index2D& u_C) const {
  // The index cast to a float is the coordinates of the lower corner of the
  // pixel.
  return u_C.cast<float>() + Vector2f(0.5f, 0.5f);
}

Index2D Lidar::imagePlaneCoordsToPixelIndex(const Vector2f& u_C) const {
  // NOTE(alexmillane): We do floor rather than a straight truncation such that
  // we handle negative image plane coordinates.
  return u_C.array().floor().cast<int>();
}

Vector3f Lidar::unprojectFromImagePlaneCoordinates(const Vector2f& u_C,
                                                   const float depth) const {
  return depth * vectorFromImagePlaneCoordinates(u_C);
}

Vector3f Lidar::unprojectFromPixelIndices(const Index2D& u_C,
                                          const float depth) const {
  return depth * vectorFromPixelIndices(u_C);
}

Vector3f Lidar::vectorFromImagePlaneCoordinates(const Vector2f& u_C) const {
  // NOTE(alexmillane): We don't do any bounds checking, i.e. that the point is
  // actually on the image plane.
  const float polar_angle_rad =
      u_C.y() * rads_per_pixel_elevation_ + start_polar_angle_rad_;
  const float azimuth_angle_rad =
      u_C.x() * rads_per_pixel_azimuth_ + start_azimuth_angle_rad_;
  return Vector3f(cos(azimuth_angle_rad) * sin(polar_angle_rad),
                  sin(azimuth_angle_rad) * sin(polar_angle_rad),
                  cos(polar_angle_rad));
}

Vector3f Lidar::vectorFromPixelIndices(const Index2D& u_C) const {
  return vectorFromImagePlaneCoordinates(
      pixelIndexToImagePlaneCoordsOfCenter(u_C));
}

AxisAlignedBoundingBox Lidar::getViewAABB(const Transform& T_L_C,
                                          const float,
                                          const float max_depth) const {
  // The AABB is a square centered at the lidars location where the height is
  // determined by the lidar FoV.
  // NOTE(alexmillane): The min depth is ignored in this function, it is a
  // parameter so it matches with camera's getViewAABB()
  AxisAlignedBoundingBox box(
      Vector3f(-max_depth, -max_depth,
               -max_depth * sin(vertical_fov_rad_ / 2.0f)),
      Vector3f(max_depth, max_depth,
               max_depth * sin(vertical_fov_rad_ / 2.0f)));
  // Translate the box to the sensor's location (note that orientation doesn't
  // matter as the lidar sees in the circle)
  box.translate(T_L_C.translation());
  return box;
}

size_t Lidar::Hash::operator()(const Lidar& lidar) const {
  // Taken from:
  // https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
  size_t az_hash = std::hash<int>()(lidar.num_azimuth_divisions_);
  size_t el_hash = std::hash<int>()(lidar.num_elevation_divisions_);
  size_t fov_hash = std::hash<float>()(lidar.vertical_fov_rad_);
  return ((az_hash ^ (el_hash << 1)) >> 1) ^ (fov_hash << 1);
}

bool operator==(const Lidar& lhs, const Lidar& rhs) {
  return (lhs.num_azimuth_divisions_ == rhs.num_azimuth_divisions_) &&
         (lhs.num_elevation_divisions_ == rhs.num_elevation_divisions_) &&
         (std::fabs(lhs.vertical_fov_rad_ - rhs.vertical_fov_rad_) <
          std::numeric_limits<float>::epsilon());
}

std::ostream& operator<<(std::ostream& os, const Lidar& lidar) {
  constexpr float kRadToDegrees = 180.0f / M_PI;
  os << "Lidar with intrinsics:\n"
     << "\tnum_azimuth_divisions: " << lidar.num_azimuth_divisions() << "\n"
     << "\tnum_elevation_divisions: " << lidar.num_elevation_divisions() << "\n"
     << "\tvertical_fov_deg: " << lidar.vertical_fov_rad() * kRadToDegrees << "\n"
     << "\tstart_polar_angle_deg: " << lidar.start_polar_angle_rad() * kRadToDegrees;
  return os;
}

}  // namespace nvblox
