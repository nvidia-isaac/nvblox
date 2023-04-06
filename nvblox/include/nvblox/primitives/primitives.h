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
#pragma once

#include "nvblox/utils/logging.h"

#include "nvblox/core/color.h"
#include "nvblox/core/types.h"

namespace nvblox {
namespace primitives {

/// Base class for primitive objects.
class Primitive {
 public:
  enum class Type { kPlane, kCube, kSphere, kCylinder };

  /// Epsilon for ray intersection and computation.
  static constexpr float kEpsilon = 1e-4;

  Primitive(const Vector3f& center, Type type)
      : Primitive(center, type, Color::White()) {}
  Primitive(const Vector3f& center, Type type, const Color& color)
      : center_(center), type_(type), color_(color) {}
  virtual ~Primitive() {}

  /// Map-building accessors.
  virtual float getDistanceToPoint(const Vector3f& point) const = 0;

  Color getColor() const { return color_; }
  Type getType() const { return type_; }

  /// Raycasting accessors.
  virtual bool getRayIntersection(const Vector3f& ray_origin,
                                  const Vector3f& ray_direction, float max_dist,
                                  Vector3f* intersect_point,
                                  float* intersect_dist) const = 0;

 protected:
  Vector3f center_;
  Type type_;
  Color color_;
};

/// Primitive sphere, given a center and a radius.
class Sphere : public Primitive {
 public:
  Sphere(const Vector3f& center, float radius)
      : Primitive(center, Type::kSphere), radius_(radius) {}
  Sphere(const Vector3f& center, float radius, const Color& color)
      : Primitive(center, Type::kSphere, color), radius_(radius) {}

  virtual float getDistanceToPoint(const Vector3f& point) const override;

  virtual bool getRayIntersection(const Vector3f& ray_origin,
                                  const Vector3f& ray_direction, float max_dist,
                                  Vector3f* intersect_point,
                                  float* intersect_dist) const override;

 protected:
  float radius_;
};

/// Primitive cube, given a center and an X,Y,Z size (can be a rectangular
/// prism).
class Cube : public Primitive {
 public:
  Cube(const Vector3f& center, const Vector3f& size)
      : Primitive(center, Type::kCube), size_(size) {}
  Cube(const Vector3f& center, const Vector3f& size, const Color& color)
      : Primitive(center, Type::kCube, color), size_(size) {}

  virtual float getDistanceToPoint(const Vector3f& point) const override;
  virtual bool getRayIntersection(const Vector3f& ray_origin,
                                  const Vector3f& ray_direction, float max_dist,
                                  Vector3f* intersect_point,
                                  float* intersect_dist) const override;

 protected:
  Vector3f size_;
};

/// Primitive plane, given a center and a normal.
/// Requires normal being passed in to ALREADY BE NORMALIZED!!!!
class Plane : public Primitive {
 public:
  Plane(const Vector3f& center, const Vector3f& normal)
      : Primitive(center, Type::kPlane), normal_(normal) {
    CHECK_NEAR(normal.norm(), 1.0, 1e-3);
  }
  Plane(const Vector3f& center, const Vector3f& normal, const Color& color)
      : Primitive(center, Type::kPlane, color), normal_(normal) {
    CHECK_NEAR(normal.norm(), 1.0, 1e-3);
  }

  virtual float getDistanceToPoint(const Vector3f& point) const override;

  virtual bool getRayIntersection(const Vector3f& ray_origin,
                                  const Vector3f& ray_direction, float max_dist,
                                  Vector3f* intersect_point,
                                  float* intersect_dist) const override;

 protected:
  Vector3f normal_;
};

/// Cylinder centered on the XY plane, with a given radius and height (in Z).
class Cylinder : public Primitive {
 public:
  Cylinder(const Vector3f& center, float radius, float height)
      : Primitive(center, Type::kCylinder), radius_(radius), height_(height) {}
  Cylinder(const Vector3f& center, float radius, float height,
           const Color& color)
      : Primitive(center, Type::kCylinder, color),
        radius_(radius),
        height_(height) {}

  virtual float getDistanceToPoint(const Vector3f& point) const override;

  virtual bool getRayIntersection(const Vector3f& ray_origin,
                                  const Vector3f& ray_direction, float max_dist,
                                  Vector3f* intersect_point,
                                  float* intersect_dist) const override;

 protected:
  float radius_;
  float height_;
};

}  // namespace primitives
}  // namespace nvblox