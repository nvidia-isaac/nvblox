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

// Image quad fragment shader - renders depth and color textures to a
// fullscreen quad. Supports multiple display layouts (side-by-side, overlay,
// single channel) and depth colormaps (grayscale, jet, turbo).

#version 450
#extension GL_GOOGLE_include_directive : enable

#include "color_utils.glsl"

layout(location = 0) in vec2 fragTexCoord;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D depthTexture;
layout(binding = 1) uniform sampler2D colorTexture;

layout(push_constant) uniform PushConstants {
    float minDepth;
    float maxDepth;
    float colormap;      // See kColormap* constants below
    float displayLayout; // See kLayout* constants below
} pc;

// Constants
const float kInvalidDepthColor = 0.2;     // Dark gray for invalid depth values
const float kRangeEpsilon = 0.0001;       // Minimum range to avoid division by zero
const float kDefaultNormalizedDepth = 0.5; // Default depth when range is zero
const float kOverlayBlendAlpha = 0.5;     // Blend factor for depth-color overlay
const float kSideBySideSplitPoint = 0.5;  // Screen split point for side-by-side layout
const float kFloatToIntRounding = 0.5;    // Added to float enum before int conversion

// Colormap enum values
const int kColormapGrayscale = 0;
const int kColormapJet = 1;
const int kColormapTurbo = 2;

// Layout enum values
const int kLayoutSideBySide = 0;
const int kLayoutColorOnly = 1;
const int kLayoutDepthOnly = 2;
const int kLayoutOverlay = 3;

vec3 applyColormap(float depth) {
    // Invalid depth (non-positive or beyond maxDepth) -> dark gray
    if (depth <= 0.0 || depth > pc.maxDepth) {
        return vec3(kInvalidDepthColor);
    }

    // Normalize depth with division-by-zero protection
    float range = pc.maxDepth - pc.minDepth;
    float t = (range > kRangeEpsilon) ? (depth - pc.minDepth) / range : kDefaultNormalizedDepth;
    t = clamp(t, 0.0, 1.0);

    // Use rounding to safely convert float enum to int (prevents truncation issues)
    int colormapType = int(pc.colormap + kFloatToIntRounding);
    if (colormapType == kColormapGrayscale) {
        return vec3(t);
    } else if (colormapType == kColormapJet) {
        return jetColormap(t);
    } else {
        // Turbo (default)
        return turboColormap(t);
    }
}

void main() {
    // Use rounding to safely convert float enum to int (prevents truncation issues)
    int layoutType = int(pc.displayLayout + kFloatToIntRounding);

    if (layoutType == kLayoutSideBySide) {
        // Side by side: depth on left, color on right
        if (fragTexCoord.x < kSideBySideSplitPoint) {
            // Depth on left half
            vec2 depthUV = vec2(fragTexCoord.x * 2.0, fragTexCoord.y);
            float depth = texture(depthTexture, depthUV).r;
            outColor = vec4(applyColormap(depth), 1.0);
        } else {
            // Color on right half
            vec2 colorUV = vec2((fragTexCoord.x - kSideBySideSplitPoint) * 2.0, fragTexCoord.y);
            outColor = texture(colorTexture, colorUV);
        }
    } else if (layoutType == kLayoutColorOnly) {
        outColor = texture(colorTexture, fragTexCoord);
    } else if (layoutType == kLayoutDepthOnly) {
        float depth = texture(depthTexture, fragTexCoord).r;
        outColor = vec4(applyColormap(depth), 1.0);
    } else {
        // Overlay: depth colormap blended with color
        float depth = texture(depthTexture, fragTexCoord).r;
        vec3 depthColor = applyColormap(depth);
        vec4 color = texture(colorTexture, fragTexCoord);

        // Blend: show depth colormap where depth is valid
        float alpha = (depth > 0.0 && depth <= pc.maxDepth) ? kOverlayBlendAlpha : 0.0;
        outColor = vec4(mix(color.rgb, depthColor, alpha), 1.0);
    }
}
