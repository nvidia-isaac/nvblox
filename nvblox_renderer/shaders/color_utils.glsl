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

// Shared color utility functions for nvblox renderer shaders.
// Include via: #include "color_utils.glsl"

#ifndef COLOR_UTILS_GLSL
#define COLOR_UTILS_GLSL

// sRGB conversion constants
// Ref: https://en.wikipedia.org/wiki/SRGB#Transfer_function_(%22gamma%22)
const float kColorNormalization = 255.0;
const float kSrgbLinearThreshold = 0.04045;
const float kSrgbLinearScale = 12.92;
const float kSrgbGamma = 2.4;

// Convert sRGB to linear color space using the full sRGB transfer function.
// Camera output is typically sRGB, but we output to sRGB swapchain which expects linear input.
vec3 srgbToLinear(vec3 srgb) {
    vec3 linearPart = srgb / kSrgbLinearScale;
    vec3 gammaPart = pow((srgb + 0.055) / 1.055, vec3(kSrgbGamma));
    return mix(linearPart, gammaPart, step(vec3(kSrgbLinearThreshold), srgb));
}

// Turbo colormap polynomial approximation.
// Ref: https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
vec3 turboColormap(float t) {
    const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
    const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
    const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
    const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
    const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
    const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

    t = clamp(t, 0.0, 1.0);
    vec4 v4 = vec4(1.0, t, t * t, t * t * t);
    vec2 v2 = v4.zw * v4.z;

    return vec3(
        dot(v4, kRedVec4) + dot(v2, kRedVec2),
        dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
        dot(v4, kBlueVec4) + dot(v2, kBlueVec2)
    );
}

// Jet colormap (piecewise linear, based on MATLAB Jet).
// Ref: https://www.mathworks.com/help/matlab/ref/jet.html#bvisv5k-1_sep_shared-m
vec3 jetColormap(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c;
    if (t < 0.25) {
        c = vec3(0.0, 4.0 * t, 1.0);
    } else if (t < 0.5) {
        c = vec3(0.0, 1.0, 1.0 - 4.0 * (t - 0.25));
    } else if (t < 0.75) {
        c = vec3(4.0 * (t - 0.5), 1.0, 0.0);
    } else {
        c = vec3(1.0, 1.0 - 4.0 * (t - 0.75), 0.0);
    }
    return c;
}

#endif // COLOR_UTILS_GLSL
