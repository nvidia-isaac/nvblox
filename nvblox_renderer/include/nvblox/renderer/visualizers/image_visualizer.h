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

#include <utility>

#include <vulkan/vulkan.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/renderer/core/shared_texture.h"
#include "nvblox/renderer/visualizers/textured_visualizer.h"

namespace nvblox {
namespace renderer {

class VkContext;
class PipelineBuilder;

/// Configuration for ImageVisualizer initialization.
/// Provides a unified way to specify image dimensions with sensible defaults.
struct ImageVisualizerConfig {
  uint32_t depth_width =
      1;  ///< Width of depth images (default: 1 for lazy init).
  uint32_t depth_height =
      1;  ///< Height of depth images (default: 1 for lazy init).
  uint32_t color_width =
      1;  ///< Width of color images (default: 1 for lazy init).
  uint32_t color_height =
      1;  ///< Height of color images (default: 1 for lazy init).

  ImageVisualizerConfig() = default;

  /// Create a config with same dimensions for both depth and color.
  /// @param width Width for both depth and color images.
  /// @param height Height for both depth and color images.
  ImageVisualizerConfig(uint32_t width, uint32_t height)
      : depth_width(width),
        depth_height(height),
        color_width(width),
        color_height(height) {}

  /// Create a config with independent depth and color dimensions.
  ImageVisualizerConfig(uint32_t depth_w, uint32_t depth_h, uint32_t color_w,
                        uint32_t color_h)
      : depth_width(depth_w),
        depth_height(depth_h),
        color_width(color_w),
        color_height(color_h) {}
};

/// Visualizer for depth and color images.
/// Renders images as textured quads with optional depth colormapping.
/// @note NOT thread-safe. See VisualizerInterface for thread safety
/// requirements.
class ImageVisualizer : public TexturedVisualizer<ImageVisualizer, 2> {
 public:
  /// Texture index for the depth image.
  static constexpr size_t kDepthTexture = 0;
  /// Texture index for the color image.
  static constexpr size_t kColorTexture = 1;

  /// Depth colormap options.
  enum class DepthColormap {
    kGrayscale,  // Black to white
    kJet,        // Blue to red (classic)
    kTurbo,      // Improved perceptual colormap
  };

  /// Display layout options.
  enum class Layout {
    kSideBySide,  // Depth on left, color on right
    kColorOnly,   // Only color image
    kDepthOnly,   // Only depth image
    kOverlay,     // Depth overlaid on color
  };

  ImageVisualizer() = default;
  ~ImageVisualizer() override;

  // Non-copyable, non-movable (prevent accidental moves of Vulkan resources)
  ImageVisualizer(const ImageVisualizer&) = delete;
  ImageVisualizer& operator=(const ImageVisualizer&) = delete;
  ImageVisualizer(ImageVisualizer&&) = delete;
  ImageVisualizer& operator=(ImageVisualizer&&) = delete;

  // VisualizerInterface methods
  /// Initialize with placeholder 1x1 textures. Call
  /// updateDepthImage/updateColorImage to set actual dimensions and data. Use
  /// init(ctx, config) if dimensions are known upfront.
  bool init(VkContext* ctx) override;
  void destroy() override;
  void render(VkCommandBuffer cmd, const ViewCamera* camera,
              uint32_t viewport_width, uint32_t viewport_height) override;

  /// Render into an explicit viewport rect (no camera needed for 2D quads).
  void render(VkCommandBuffer cmd, uint32_t x, uint32_t y, uint32_t width,
              uint32_t height);

  bool hasData() const override {
    return texture(kDepthTexture) || texture(kColorTexture);
  }

  /// Initialize the visualizer with a configuration struct.
  /// @param ctx Vulkan context.
  /// @param config Configuration specifying image dimensions.
  /// @return True if initialization succeeded.
  bool init(VkContext* ctx, const ImageVisualizerConfig& config);

  /// Update depth image from CUDA memory.
  /// @param depth_ptr CUDA device pointer to float depth data (contiguous).
  /// @param stream CUDA stream.
  void updateDepth(const float* depth_ptr, const CudaStream& stream) {
    copyToTexture(kDepthTexture, depth_ptr, stream);
  }

  /// Update color image from CUDA memory.
  /// @param color_ptr CUDA device pointer to RGBA8 color data (contiguous).
  /// @param stream CUDA stream.
  void updateColor(const void* color_ptr, const CudaStream& stream) {
    copyToTexture(kColorTexture, color_ptr, stream);
  }

  /// Set depth colormap.
  void setDepthColormap(DepthColormap colormap) { colormap_ = colormap; }

  /// Set display layout.
  void setLayout(Layout layout) { layout_ = layout; }

  /// Set depth range for colormap normalization.
  void setDepthRange(float min_depth, float max_depth) {
    min_depth_ = min_depth;
    max_depth_ = max_depth;
  }

  /// Resize depth texture if dimensions differ.
  /// @param width New width.
  /// @param height New height.
  /// @return True if resize succeeded or not needed.
  bool resizeDepthTexture(uint32_t width, uint32_t height) {
    return resizeTexture(kDepthTexture, width, height);
  }

  /// Resize color texture if dimensions differ.
  /// @param width New width.
  /// @param height New height.
  /// @return True if resize succeeded or not needed.
  bool resizeColorTexture(uint32_t width, uint32_t height) {
    return resizeTexture(kColorTexture, width, height);
  }

  /// Get depth texture width.
  uint32_t depthWidth() const {
    auto* tex = texture(kDepthTexture);
    return tex ? tex->width() : 0;
  }
  /// Get depth texture height.
  uint32_t depthHeight() const {
    auto* tex = texture(kDepthTexture);
    return tex ? tex->height() : 0;
  }

  /// Get color texture width.
  uint32_t colorWidth() const {
    auto* tex = texture(kColorTexture);
    return tex ? tex->width() : 0;
  }
  /// Get color texture height.
  uint32_t colorHeight() const {
    auto* tex = texture(kColorTexture);
    return tex ? tex->height() : 0;
  }

  /// @cond INTERNAL
  // =========================================================================
  // TexturedVisualizer template requirements
  // =========================================================================

  /// Get texture format for each texture index.
  static SharedTexture::Format textureFormat(size_t index) {
    return index == kDepthTexture ? SharedTexture::Format::kR32F
                                  : SharedTexture::Format::kRGB8;
  }

  /// Get descriptor binding for each texture index.
  static TextureBinding textureBinding(size_t index) {
    return {static_cast<uint32_t>(index), VK_SHADER_STAGE_FRAGMENT_BIT};
  }

  /// Get default texture dimensions for each texture index.
  static std::pair<uint32_t, uint32_t> defaultTextureDimensions(size_t) {
    return {1, 1};  // Placeholder, resized on first update
  }

  /// Shader base name.
  static constexpr const char* shaderName() { return "image_quad"; }

  /// Configure pipeline vertex attributes and topology.
  void configurePipeline(PipelineBuilder& builder) const;

  /// Push constant size in bytes.
  static constexpr uint32_t pushConstantSize();

  /// Shader stages using push constants.
  static constexpr VkShaderStageFlags pushConstantStages() {
    return VK_SHADER_STAGE_FRAGMENT_BIT;
  }

  /// Visualizer name for logging.
  static constexpr const char* visualizerName() { return "ImageVisualizer"; }
  /// @endcond

 private:
  bool createPipeline();

  // Pipeline
  VkPipeline pipeline_ = VK_NULL_HANDLE;

  // Settings
  DepthColormap colormap_ = DepthColormap::kTurbo;
  Layout layout_ = Layout::kSideBySide;
  float min_depth_ = 0.1f;
  float max_depth_ = 5.0f;
};

}  // namespace renderer
}  // namespace nvblox
