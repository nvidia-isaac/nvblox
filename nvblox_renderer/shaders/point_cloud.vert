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

// Point cloud vertex shader - transforms point positions by the view-projection
// matrix, sets configurable point size, and converts colors from sRGB uint8 to
// linear float.

#version 450
#extension GL_GOOGLE_include_directive : enable

#include "color_utils.glsl"

// Vertex input: interleaved position (vec3) + color (RGBA as uvec4)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in uvec4 inColor;  // VK_FORMAT_R8G8B8A8_UINT produces uvec4

// Output to fragment shader
layout(location = 0) out vec4 fragColor;

// Push constants
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    float pointSize;
} pc;

void main() {
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);
    gl_PointSize = pc.pointSize;

    // Convert RGBA uint8 to normalized float and apply sRGB to linear conversion
    vec4 color = vec4(inColor) / kColorNormalization;
    fragColor = vec4(srgbToLinear(color.rgb), color.a);
}
