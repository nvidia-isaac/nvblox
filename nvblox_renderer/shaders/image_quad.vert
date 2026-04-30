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

// Image quad vertex shader - generates a fullscreen quad procedurally
// (no vertex buffer needed). Passes texture coordinates to the fragment shader.

#version 450

layout(location = 0) out vec2 fragTexCoord;

void main() {
    // Generate fullscreen quad vertices
    // Triangle 1: (0,1,2), Triangle 2: (2,1,3)
    // Vertices: 0=(-1,-1), 1=(1,-1), 2=(-1,1), 3=(1,1)

    vec2 positions[6] = vec2[](
        vec2(-1.0, -1.0),  // 0
        vec2( 1.0, -1.0),  // 1
        vec2(-1.0,  1.0),  // 2
        vec2(-1.0,  1.0),  // 2
        vec2( 1.0, -1.0),  // 1
        vec2( 1.0,  1.0)   // 3
    );

    vec2 texCoords[6] = vec2[](
        vec2(0.0, 0.0),
        vec2(1.0, 0.0),
        vec2(0.0, 1.0),
        vec2(0.0, 1.0),
        vec2(1.0, 0.0),
        vec2(1.0, 1.0)
    );

    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragTexCoord = texCoords[gl_VertexIndex];
}
