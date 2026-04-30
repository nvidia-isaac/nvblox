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

// Mesh vertex shader - transforms mesh vertices by the view-projection matrix
// and converts vertex colors from sRGB uint8 to linear float.

#version 450
#extension GL_GOOGLE_include_directive : enable

#include "color_utils.glsl"

// Vertex input: interleaved position (vec3) + color (RGBA as uvec4) + UV (vec2)
layout(location = 0) in vec3 inPosition;
layout(location = 1) in uvec4 inColor;  // VK_FORMAT_R8G8B8A8_UINT produces uvec4
layout(location = 2) in vec2 inUV;      // UV coordinates (-1,-1 = no texture)

// Output to fragment shader
layout(location = 0) out vec4 fragColor;
layout(location = 1) out vec2 fragUV;

// Push constants
layout(push_constant) uniform PushConstants {
    mat4 viewProj;
    uint hasTexture;  // 0 = vertex color only, 1 = texture atlas bound
} pc;

void main() {
    gl_Position = pc.viewProj * vec4(inPosition, 1.0);

    // Convert RGBA uint8 to normalized float and apply sRGB to linear conversion
    vec4 color = vec4(inColor) / kColorNormalization;
    fragColor = vec4(srgbToLinear(color.rgb), color.a);

    // Pass UV to fragment shader
    fragUV = inUV;
}
