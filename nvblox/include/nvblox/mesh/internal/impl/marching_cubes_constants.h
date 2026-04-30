/*
Copyright 2026 NVIDIA CORPORATION

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

namespace nvblox {
namespace marching_cubes {

/// Structural constants of the marching cubes algorithm.
/// A cube has 8 corners, 12 edges, and each triangle has 3 vertices.
/// The 8-bit corner sign pattern yields 256 possible configurations;
/// configurations 0 (all outside) and 255 (all inside) produce no surface.
constexpr int kNumCorners = 8;
constexpr int kNumEdges = 12;
constexpr int kVerticesPerTriangle = 3;
constexpr int kNumConfigurations = 256;
constexpr int kAllInsideConfig = kNumConfigurations - 1;

}  // namespace marching_cubes
}  // namespace nvblox
