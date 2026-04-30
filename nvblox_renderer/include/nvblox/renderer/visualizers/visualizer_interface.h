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

#include <vulkan/vulkan.h>

namespace nvblox {
namespace renderer {

class VkContext;
class ViewCamera;

/// Visualizer lifecycle state.
/// Used to query whether a visualizer is ready for rendering.
enum class VisualizerState {
  kUninitialized,  ///< Not yet initialized or destroyed.
  kReady,          ///< Successfully initialized and ready to render.
  kError           ///< Initialization failed or in error state.
};

/// Base interface for all visualizers.
/// Visualizers are responsible for rendering specific types of data
/// (images, point clouds, meshes, etc.)
///
/// THREAD SAFETY: Visualizers are NOT thread-safe. All methods (init, destroy,
/// render, update*, hasData) must be called from a single thread, or externally
/// synchronized by the caller. Concurrent calls to update*() and render() will
/// cause data races and undefined behavior.
class VisualizerInterface {
 public:
  virtual ~VisualizerInterface() = default;

  /// Initialize the visualizer.
  /// @param ctx Vulkan context.
  /// @return True if initialization succeeded.
  virtual bool init(VkContext* ctx) = 0;

  /// Destroy resources.
  virtual void destroy() = 0;

  /// Record rendering commands.
  /// @param cmd Command buffer to record into.
  /// @param camera View camera (may be null for 2D visualizers).
  /// @param viewport_width Viewport width in pixels.
  /// @param viewport_height Viewport height in pixels.
  virtual void render(VkCommandBuffer cmd, const ViewCamera* camera,
                      uint32_t viewport_width, uint32_t viewport_height) = 0;

  /// Check if the visualizer has data to render.
  virtual bool hasData() const = 0;
};

}  // namespace renderer
}  // namespace nvblox
