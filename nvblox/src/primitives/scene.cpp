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
#include "nvblox/primitives/scene.h"

namespace nvblox {
namespace primitives {

Scene::Scene() : aabb_(Vector3f(-5.0, -5.0, -1.0), Vector3f(5.0, 5.0, 9.0)) {}

void Scene::addPrimitive(std::unique_ptr<Primitive> primitive) {
  primitives_.emplace_back(std::move(primitive));
}

void Scene::addGroundLevel(float height) {
  primitives_.emplace_back(
      new Plane(Vector3f(0.0, 0.0, height), Vector3f(0.0, 0.0, 1.0)));
}

void Scene::addCeiling(float height) {
  primitives_.emplace_back(
      new Plane(Vector3f(0.0, 0.0, height), Vector3f(0.0, 0.0, -1.0)));
}

void Scene::addPlaneBoundaries(float x_min, float x_max, float y_min,
                               float y_max) {
  // X planes:
  primitives_.emplace_back(
      new Plane(Vector3f(x_min, 0.0, 0.0), Vector3f(1.0, 0.0, 0.0)));
  primitives_.emplace_back(
      new Plane(Vector3f(x_max, 0.0, 0.0), Vector3f(-1.0, 0.0, 0.0)));

  // Y planes:
  primitives_.emplace_back(
      new Plane(Vector3f(0.0, y_min, 0.0), Vector3f(0.0, 1.0, 0.0)));
  primitives_.emplace_back(
      new Plane(Vector3f(0.0, y_max, 0.0), Vector3f(0.0, -1.0, 0.0)));
}

void Scene::clear() { primitives_.clear(); }

float Scene::getSignedDistanceToPoint(const Vector3f& coords,
                                      float max_dist) const {
  float min_dist = max_dist;
  for (const std::unique_ptr<Primitive>& primitive : primitives_) {
    float primitive_dist = primitive->getDistanceToPoint(coords);
    if (primitive_dist < min_dist) {
      min_dist = primitive_dist;
    }
  }

  return min_dist;
}

bool Scene::getRayIntersection(const Vector3f& ray_origin,
                               const Vector3f& ray_direction, float max_dist,
                               Vector3f* ray_intersection,
                               float* ray_dist) const {
  CHECK_NOTNULL(ray_intersection);
  CHECK_NOTNULL(ray_dist);
  *ray_intersection = Vector3f::Zero();
  *ray_dist = max_dist;
  bool ray_valid = false;
  for (const std::unique_ptr<Primitive>& primitive : primitives_) {
    Vector3f primitive_intersection;
    float primitive_dist;
    bool intersects =
        primitive->getRayIntersection(ray_origin, ray_direction, max_dist,
                                      &primitive_intersection, &primitive_dist);
    if (intersects) {
      if (!ray_valid || primitive_dist < *ray_dist) {
        ray_valid = true;
        *ray_dist = primitive_dist;
        *ray_intersection = primitive_intersection;
      }
    }
  }

  return ray_valid;
}

void Scene::generateDepthImageFromScene(const Camera& camera,
                                        const Transform& T_S_C, float max_dist,
                                        DepthImage* depth_frame) const {
  CHECK_NOTNULL(depth_frame);
  CHECK(depth_frame->memory_type() == MemoryType::kUnified)
      << "For scene generation DepthImage with memory_type == kUnified is "
         "required.";
  CHECK_EQ(depth_frame->rows(), camera.height());
  CHECK_EQ(depth_frame->cols(), camera.width());

  const Transform T_C_S = T_S_C.inverse();

  // Iterate over the entire image.
  Index2D u_C;
  for (u_C.x() = 0; u_C.x() < camera.width(); u_C.x()++) {
    for (u_C.y() = 0; u_C.y() < camera.height(); u_C.y()++) {
      // Get the ray going through this pixel.
      const Vector3f ray_direction =
          T_S_C.linear() * camera.vectorFromPixelIndices(u_C).normalized();
      // Get the intersection point for this ray.
      Vector3f ray_intersection;
      float ray_dist;
      if (getRayIntersection(T_S_C.translation(), ray_direction, max_dist,
                             &ray_intersection, &ray_dist)) {
        // The ray intersection is expressed in the world coordinate frame.
        // We must transform it back to the camera coordinate frame.
        const Vector3f p_C = T_C_S * ray_intersection;
        // Then we use the z coordinate in the camera frame to set the depth.
        (*depth_frame)(u_C.y(), u_C.x()) = p_C.z();
      } else {
        // Otherwise set the depth to 0.0 to mark it as invalid.
        (*depth_frame)(u_C.y(), u_C.x()) = 0.0f;
      }
    }
  }
}

}  // namespace primitives
}  // namespace nvblox