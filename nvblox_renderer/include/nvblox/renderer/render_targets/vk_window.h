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

#include <functional>
#include <string>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

namespace nvblox {
namespace renderer {

/// Common callback types for window events.
using WindowResizeCallback = std::function<void(int width, int height)>;
using WindowKeyCallback = std::function<void(int key, int action, int mods)>;
using WindowMouseButtonCallback =
    std::function<void(int button, int action, int mods)>;
using WindowMouseMoveCallback = std::function<void(double x, double y)>;
using WindowScrollCallback =
    std::function<void(double x_offset, double y_offset)>;

/// Abstract interface for window backends.
/// Allows different windowing systems (GLFW, SDL, X11, etc.) to be used
/// with the renderer. Implementations handle window creation, event polling,
/// and Vulkan surface creation.
class IWindow {
 public:
  virtual ~IWindow() = default;

  /// Get required Vulkan instance extensions for this window backend.
  /// @return Vector of extension names required by this backend.
  virtual std::vector<const char*> getRequiredInstanceExtensions() const = 0;

  /// Create a window.
  /// @param width Initial width in pixels.
  /// @param height Initial height in pixels.
  /// @param title Window title.
  /// @return True if window creation succeeded.
  virtual bool create(int width, int height, const std::string& title) = 0;

  /// Create Vulkan surface for this window.
  /// @param instance Vulkan instance.
  /// @param surface Output: created surface.
  /// @return True if surface creation succeeded.
  virtual bool createSurface(VkInstance instance, VkSurfaceKHR* surface) = 0;

  /// Poll window events.
  virtual void pollEvents() = 0;

  /// Check if window should close.
  virtual bool shouldClose() const = 0;

  /// Get current window size.
  virtual void getSize(int* width, int* height) const = 0;

  /// Get framebuffer size (may differ from window size on HiDPI).
  virtual void getFramebufferSize(int* width, int* height) const = 0;

  /// Check if window is minimized.
  virtual bool isMinimized() const = 0;

  /// Resize the window.
  virtual void resize(int width, int height) = 0;

  /// Set callback for window resize.
  virtual void setResizeCallback(WindowResizeCallback callback) = 0;

  /// Set callback for key events.
  virtual void setKeyCallback(WindowKeyCallback callback) = 0;

  /// Set callback for mouse button events.
  virtual void setMouseButtonCallback(WindowMouseButtonCallback callback) = 0;

  /// Set callback for mouse move events.
  virtual void setMouseMoveCallback(WindowMouseMoveCallback callback) = 0;

  /// Set callback for scroll events.
  virtual void setScrollCallback(WindowScrollCallback callback) = 0;

  /// Get current cursor position.
  virtual void getCursorPos(double* x, double* y) const = 0;

  /// Check if a mouse button is pressed.
  virtual bool isMouseButtonPressed(int button) const = 0;
};

/// GLFW window implementation for Vulkan rendering.
class VkWindow : public IWindow {
 public:
  // Type aliases for backward compatibility
  using ResizeCallback = WindowResizeCallback;
  using KeyCallback = WindowKeyCallback;
  using MouseButtonCallback = WindowMouseButtonCallback;
  using MouseMoveCallback = WindowMouseMoveCallback;
  using ScrollCallback = WindowScrollCallback;

  VkWindow() = default;
  ~VkWindow() override;

  // Non-copyable
  VkWindow(const VkWindow&) = delete;
  VkWindow& operator=(const VkWindow&) = delete;

  /// Get required Vulkan instance extensions for GLFW window presentation.
  /// Call this before creating the VkContext to get the extensions needed.
  /// Static version for use before window is created.
  /// @return Vector of extension names required by GLFW.
  static std::vector<const char*> getRequiredVulkanExtensions();

  // IWindow interface
  std::vector<const char*> getRequiredInstanceExtensions() const override;
  bool create(int width, int height, const std::string& title) override;
  bool createSurface(VkInstance instance, VkSurfaceKHR* surface) override;
  void pollEvents() override;
  bool shouldClose() const override;
  void getSize(int* width, int* height) const override;
  void getFramebufferSize(int* width, int* height) const override;
  bool isMinimized() const override;
  void resize(int width, int height) override;
  void setResizeCallback(WindowResizeCallback callback) override;
  void setKeyCallback(WindowKeyCallback callback) override;
  void setMouseButtonCallback(WindowMouseButtonCallback callback) override;
  void setMouseMoveCallback(WindowMouseMoveCallback callback) override;
  void setScrollCallback(WindowScrollCallback callback) override;
  void getCursorPos(double* x, double* y) const override;
  bool isMouseButtonPressed(int button) const override;

  /// Get the underlying GLFW window handle (GLFW-specific).
  GLFWwindow* handle() const { return window_; }

 private:
  static void framebufferResizeCallback(GLFWwindow* window, int width,
                                        int height);
  static void keyCallback(GLFWwindow* window, int key, int scancode, int action,
                          int mods);
  static void mouseButtonCallback(GLFWwindow* window, int button, int action,
                                  int mods);
  static void cursorPosCallback(GLFWwindow* window, double x, double y);
  static void scrollCallback(GLFWwindow* window, double x_offset,
                             double y_offset);

  GLFWwindow* window_ = nullptr;
  ResizeCallback resize_callback_;
  KeyCallback key_callback_;
  MouseButtonCallback mouse_button_callback_;
  MouseMoveCallback mouse_move_callback_;
  ScrollCallback scroll_callback_;
  bool framebuffer_resized_ = false;
};

}  // namespace renderer
}  // namespace nvblox
