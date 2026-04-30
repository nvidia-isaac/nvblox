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

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <glog/logging.h>

namespace nvblox {
namespace renderer {

// =============================================================================
// Rendering Constants
// =============================================================================

/// Default clear color for rendering (dark gray).
constexpr std::array<float, 4> kDefaultClearColor = {0.1f, 0.1f, 0.1f, 1.0f};

constexpr float kDefaultDepthClear = 1.0f;  ///< Far plane.

// =============================================================================
// Buffer Size Constants
// =============================================================================

// Initial Vulkan buffer sizes. Kept small to leave GPU memory for nvblox
// (assumes ~8GB VRAM shared with nvblox). Buffers grow automatically when
// the data exceeds these sizes, so these only affect the first allocation.

/// Default initial buffer size for point cloud visualizer (in points).
constexpr size_t kDefaultPointBufferSize = 100'000;

/// Default initial buffer size for mesh vertices.
constexpr size_t kDefaultVertexBufferSize = 100'000;

/// Default initial buffer size for mesh indices (~100k triangles).
constexpr size_t kDefaultIndexBufferSize = 300'000;

/// Growth factor when resizing buffers (1.5x provides good amortized cost).
constexpr float kBufferGrowthFactor = 1.5f;

/// Maximum allowed buffer size in bytes (1 GB).
constexpr size_t kMaxBufferSizeBytes = 1024ULL * 1024ULL * 1024ULL;

constexpr size_t kMaxVertexCount = 50'000'000;    // 50M
constexpr size_t kMaxTriangleCount = 50'000'000;  // 50M
constexpr size_t kMaxPointCount = 50'000'000;     // 50M

/// Check if a * b would overflow size_t.
constexpr bool wouldOverflow(size_t a, size_t b) {
  if (a == 0 || b == 0) return false;
  return a > std::numeric_limits<size_t>::max() / b;
}

/// Calculate new buffer capacity with growth factor, capped at
/// kMaxBufferSizeBytes.
inline size_t calculateResizeCapacity(size_t required_size) {
  size_t new_size = static_cast<size_t>(required_size * kBufferGrowthFactor);
  // Cap at maximum to prevent excessive allocation
  return (new_size > kMaxBufferSizeBytes) ? kMaxBufferSizeBytes : new_size;
}

struct BufferValidationResult {
  bool valid;
  size_t required_size;
};

/// Validate element count against limits and check for overflow.
inline BufferValidationResult validateBufferSize(size_t count,
                                                 size_t element_size,
                                                 size_t max_count,
                                                 const char* type_name) {
  if (count > max_count) {
    LOG(ERROR) << "update" << type_name << ": count (" << count
               << ") exceeds maximum (" << max_count << ")";
    return {false, 0};
  }
  if (wouldOverflow(count, element_size)) {
    LOG(ERROR) << "update" << type_name << ": buffer size would overflow";
    return {false, 0};
  }
  size_t required = count * element_size;
  if (required > kMaxBufferSizeBytes) {
    LOG(ERROR) << "update" << type_name << ": required size (" << required
               << ") exceeds maximum (" << kMaxBufferSizeBytes << ")";
    return {false, 0};
  }
  return {true, required};
}

// =============================================================================
// Camera Constants
// =============================================================================

constexpr float kDefaultCameraDistanceM = 3.0f;
constexpr float kCameraOrbitSpeedRadPerPx = 0.005f;
constexpr float kCameraPanSpeedMPerPx = 0.01f;  ///< Scaled by distance.
constexpr float kCameraResetTargetZM = 1.5f;
constexpr float kCameraResetYawRad =
    static_cast<float>(M_PI);  ///< PI = behind, looking forward.
constexpr float kCameraResetPitchRad = 0.3f;

// =============================================================================
// Timeout Constants
// =============================================================================

/// Timeout for swapchain image acquisition in nanoseconds (1 second).
constexpr uint64_t kAcquireImageTimeoutNs = 1'000'000'000;

// =============================================================================
// Texture Constants
// =============================================================================

constexpr uint32_t kMaxTextureDimension = 16384;
constexpr uint32_t kMinTextureDimension = 1;

// =============================================================================
// CUDA Constants
// =============================================================================

constexpr uint8_t kOpaqueAlpha = 255;

}  // namespace renderer
}  // namespace nvblox
