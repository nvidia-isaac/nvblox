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

// Mesh fragment shader - outputs interpolated vertex colors, discarding
// fully transparent fragments.

#version 450

layout(location = 0) in vec4 fragColor;
layout(location = 1) in vec2 fragUV;
layout(location = 0) out vec4 outColor;

// Texture atlas (only bound when hasTexture == 1)
layout(set = 0, binding = 0) uniform sampler2D texAtlas;

// Push constants
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    uint hasTexture;  // 0 = vertex color only, 1 = texture atlas bound
} pc;

// Constants
const float kTransparencyThreshold = 0.01;  // Discard fragments below this alpha

void main() {
    vec4 color;

    // Use texture if available and vertex has valid UV (>= 0)
    if (pc.hasTexture == 1u && fragUV.x >= 0.0 && fragUV.y >= 0.0) {
        color = texture(texAtlas, fragUV);
    } else {
        color = fragColor;
    }

    // Discard fully transparent fragments
    if (color.a < kTransparencyThreshold) {
        discard;
    }

    outColor = color;
}
