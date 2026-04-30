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

// Point cloud fragment shader - renders points as circles (discarding pixels
// outside the radius) and discards fully transparent fragments.

#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 0) out vec4 outColor;

// Constants
const float kTransparencyThreshold = 0.01;  // Discard fragments below this alpha
const float kPointCenter = 0.5;             // gl_PointCoord center offset
const float kCircularPointRadius = 0.5;     // Radius for circular point rendering

void main() {
    // Discard transparent points
    if (fragColor.a < kTransparencyThreshold) {
        discard;
    }

    // Make points circular instead of square
    vec2 coord = gl_PointCoord - vec2(kPointCenter);
    if (length(coord) > kCircularPointRadius) {
        discard;
    }

    outColor = fragColor;
}
