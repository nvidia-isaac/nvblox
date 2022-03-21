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

---- Original voxblox license, which this file is heavily based on: ----
Copyright (c) 2016, ETHZ ASL
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of voxblox nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include "nvblox/primitives/primitives.h"

namespace nvblox {
namespace primitives {

float Sphere::getDistanceToPoint(const Vector3f& point) const {
  float distance = (center_ - point).norm() - radius_;
  return distance;
}

bool Sphere::getRayIntersection(const Vector3f& ray_origin,
                                const Vector3f& ray_direction, float max_dist,
                                Vector3f* intersect_point,
                                float* intersect_dist) const {
  // From https://en.wikipedia.org/wiki/Line%E2%80%93sphere_intersection
  // x = o + dl is the ray equation
  // r = sphere radius, c = sphere center
  float under_square_root = pow(ray_direction.dot(ray_origin - center_), 2) -
                            (ray_origin - center_).squaredNorm() +
                            pow(radius_, 2);

  // No real roots = no intersection.
  if (under_square_root < 0.0) {
    return false;
  }

  float d =
      -(ray_direction.dot(ray_origin - center_)) - sqrt(under_square_root);

  // Intersection behind the origin.
  if (d < 0.0) {
    return false;
  }
  // Intersection greater than max dist, so no intersection in the sensor
  // range.
  if (d > max_dist) {
    return false;
  }

  *intersect_point = ray_origin + d * ray_direction;
  *intersect_dist = d;
  return true;
}

float Cube::getDistanceToPoint(const Vector3f& point) const {
  // Solution from http://stackoverflow.com/questions/5254838/
  // calculating-distance-between-a-point-and-a-rectangular-box-nearest-point

  Vector3f distance_vector = Vector3f::Zero();
  distance_vector.x() =
      std::max(std::max(center_.x() - size_.x() / 2.0 - point.x(), 0.0),
               point.x() - center_.x() - size_.x() / 2.0);
  distance_vector.y() =
      std::max(std::max(center_.y() - size_.y() / 2.0 - point.y(), 0.0),
               point.y() - center_.y() - size_.y() / 2.0);
  distance_vector.z() =
      std::max(std::max(center_.z() - size_.z() / 2.0 - point.z(), 0.0),
               point.z() - center_.z() - size_.z() / 2.0);

  float distance = distance_vector.norm();

  // Basically 0... Means it's inside!
  if (distance < kEpsilon) {
    distance_vector.x() = std::max(center_.x() - size_.x() / 2.0 - point.x(),
                                   point.x() - center_.x() - size_.x() / 2.0);
    distance_vector.y() = std::max(center_.y() - size_.y() / 2.0 - point.y(),
                                   point.y() - center_.y() - size_.y() / 2.0);
    distance_vector.z() = std::max(center_.z() - size_.z() / 2.0 - point.z(),
                                   point.z() - center_.z() - size_.z() / 2.0);
    distance = distance_vector.maxCoeff();
  }

  return distance;
}

bool Cube::getRayIntersection(const Vector3f& ray_origin,
                              const Vector3f& ray_direction, float max_dist,
                              Vector3f* intersect_point,
                              float* intersect_dist) const {
  // Adapted from https://www.scratchapixel.com/lessons/3d-basic-rendering/
  // minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
  // Compute min and max limits in 3D.

  // Precalculate signs and inverse directions.
  Vector3f inv_dir(1.0 / ray_direction.x(), 1.0 / ray_direction.y(),
                   1.0 / ray_direction.z());
  Eigen::Vector3i ray_sign(inv_dir.x() < 0.0, inv_dir.y() < 0.0,
                           inv_dir.z() < 0.0);

  Vector3f bounds[2];
  bounds[0] = center_ - size_ / 2.0;
  bounds[1] = center_ + size_ / 2.0;

  float tmin = (bounds[ray_sign.x()].x() - ray_origin.x()) * inv_dir.x();
  float tmax = (bounds[1 - ray_sign.x()].x() - ray_origin.x()) * inv_dir.x();
  float tymin = (bounds[ray_sign.y()].y() - ray_origin.y()) * inv_dir.y();
  float tymax = (bounds[1 - ray_sign.y()].y() - ray_origin.y()) * inv_dir.y();

  if ((tmin > tymax) || (tymin > tmax)) return false;
  if (tymin > tmin) tmin = tymin;
  if (tymax < tmax) tmax = tymax;

  float tzmin = (bounds[ray_sign.z()].z() - ray_origin.z()) * inv_dir.z();
  float tzmax = (bounds[1 - ray_sign.z()].z() - ray_origin.z()) * inv_dir.z();

  if ((tmin > tzmax) || (tzmin > tmax)) return false;
  if (tzmin > tmin) tmin = tzmin;
  if (tzmax < tmax) tmax = tzmax;

  float t = tmin;
  if (t < 0.0) {
    t = tmax;
    if (t < 0.0) {
      return false;
    }
  }

  if (t > max_dist) {
    return false;
  }

  *intersect_dist = t;
  *intersect_point = ray_origin + ray_direction * t;

  return true;
}

float Plane::getDistanceToPoint(const Vector3f& point) const {
  // Compute the 'd' in ax + by + cz + d = 0:
  // This is actually the scalar product I guess.
  float d = -normal_.dot(center_);
  float p = d / normal_.norm();

  float distance = normal_.dot(point) + p;
  return distance;
}

bool Plane::getRayIntersection(const Vector3f& ray_origin,
                               const Vector3f& ray_direction, float max_dist,
                               Vector3f* intersect_point,
                               float* intersect_dist) const {
  // From https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
  // Following notation of sphere more...
  // x = o + dl is the ray equation
  // n = normal, c = plane 'origin'
  float denominator = ray_direction.dot(normal_);
  if (std::abs(denominator) < kEpsilon) {
    // Lines are parallel, no intersection.
    return false;
  }
  float d = (center_ - ray_origin).dot(normal_) / denominator;
  if (d < 0.0) {
    return false;
  }
  if (d > max_dist) {
    return false;
  }
  *intersect_point = ray_origin + d * ray_direction;
  *intersect_dist = d;
  return true;
}

float Cylinder::getDistanceToPoint(const Vector3f& point) const {
  // From: https://math.stackexchange.com/questions/2064745/
  // 3 cases, depending on z of point.
  // First case: in plane with the cylinder. This also takes care of inside
  // case.
  float distance = 0.0;

  float min_z_limit = center_.z() - height_ / 2.0;
  float max_z_limit = center_.z() + height_ / 2.0;
  if (point.z() >= min_z_limit && point.z() <= max_z_limit) {
    distance = (point.head<2>() - center_.head<2>()).norm() - radius_;
  } else if (point.z() > max_z_limit) {
    // Case 2: above the cylinder.
    distance =
        std::sqrt(std::max((point.head<2>() - center_.head<2>()).squaredNorm() -
                               radius_ * radius_,
                           static_cast<float>(0.0)) +
                  (point.z() - max_z_limit) * (point.z() - max_z_limit));
  } else {
    // Case 3: below cylinder.
    distance =
        std::sqrt(std::max((point.head<2>() - center_.head<2>()).squaredNorm() -
                               radius_ * radius_,
                           static_cast<float>(0.0)) +
                  (point.z() - min_z_limit) * (point.z() - min_z_limit));
  }
  return distance;
}

bool Cylinder::getRayIntersection(const Vector3f& ray_origin,
                                  const Vector3f& ray_direction, float max_dist,
                                  Vector3f* intersect_point,
                                  float* intersect_dist) const {
  // From http://woo4.me/wootracer/cylinder-intersection/
  // and http://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node2.html
  // Define ray as P = E + tD, where E is ray_origin and D is ray_direction.
  // We define our cylinder as centered in the xy coordinate system, so
  // E in this case is actually ray_origin - center_.
  Vector3f vector_E = ray_origin - center_;
  Vector3f vector_D = ray_direction;  // Axis aligned.

  float a = vector_D.x() * vector_D.x() + vector_D.y() * vector_D.y();
  float b = 2 * vector_E.x() * vector_D.x() + 2 * vector_E.y() * vector_D.y();
  float c = vector_E.x() * vector_E.x() + vector_E.y() * vector_E.y() -
            radius_ * radius_;

  // t = (-b +- sqrt(b^2 - 4ac))/2a
  // t only has solutions if b^2 - 4ac >= 0
  float t1 = -1.0;
  float t2 = -1.0;

  // Make sure we don't divide by 0.
  if (std::abs(a) < kEpsilon) {
    return false;
  }

  float under_square_root = b * b - 4 * a * c;
  if (under_square_root < 0) {
    return false;
  }
  if (under_square_root <= kEpsilon) {
    t1 = -b / (2 * a);
    // Just keep t2 at invalid default value.
  } else {
    // 2 ts.
    t1 = (-b + std::sqrt(under_square_root)) / (2 * a);
    t2 = (-b - std::sqrt(under_square_root)) / (2 * a);
  }

  // Great, now we got some ts, time to figure out whether we hit the cylinder
  // or the endcaps.
  float t = max_dist;

  float z1 = vector_E.z() + t1 * vector_D.z();
  float z2 = vector_E.z() + t2 * vector_D.z();
  bool t1_valid = t1 >= 0.0 && z1 >= -height_ / 2.0 && z1 <= height_ / 2.0;
  bool t2_valid = t2 >= 0.0 && z2 >= -height_ / 2.0 && z2 <= height_ / 2.0;

  // Get the endcaps and their validity.
  // Check end-cap intersections now... :(
  float t3, t4;
  bool t3_valid = false, t4_valid = false;

  // Make sure we don't divide by 0.
  if (std::abs(vector_D.z()) > kEpsilon) {
    // t3 is the bottom end-cap, t4 is the top.
    t3 = (-height_ / 2.0 - vector_E.z()) / vector_D.z();
    t4 = (height_ / 2.0 - vector_E.z()) / vector_D.z();

    Vector3f q3 = vector_E + t3 * vector_D;
    Vector3f q4 = vector_E + t4 * vector_D;

    t3_valid = t3 >= 0.0 && q3.head<2>().norm() < radius_;
    t4_valid = t4 >= 0.0 && q4.head<2>().norm() < radius_;
  }

  if (!(t1_valid || t2_valid || t3_valid || t4_valid)) {
    return false;
  }
  if (t1_valid) {
    t = std::min(t, t1);
  }
  if (t2_valid) {
    t = std::min(t, t2);
  }
  if (t3_valid) {
    t = std::min(t, t3);
  }
  if (t4_valid) {
    t = std::min(t, t4);
  }

  // Intersection greater than max dist, so no intersection in the sensor
  // range.
  if (t >= max_dist) {
    return false;
  }

  // Back to normal coordinates now.
  *intersect_point = ray_origin + t * ray_direction;
  *intersect_dist = t;
  return true;
}

}  // namespace primitives
}  // namespace nvblox