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

#include "nvblox/renderer/render_targets/vk_window.h"

#include <glog/logging.h>

#include "nvblox/renderer/core/error_check.h"

namespace nvblox {
namespace renderer {

VkWindow::~VkWindow() {
  if (window_ != nullptr) {
    glfwDestroyWindow(window_);
    window_ = nullptr;
  }
  glfwTerminate();
}

// Static version - can be called before window is created
std::vector<const char*> VkWindow::getRequiredVulkanExtensions() {
  // Ensure GLFW is initialized (needed to query extensions)
  if (!glfwInit()) {
    LOG(ERROR) << "Failed to initialize GLFW for extension query";
    return {};
  }

  uint32_t count = 0;
  const char** exts = glfwGetRequiredInstanceExtensions(&count);
  if (!exts) {
    LOG(ERROR) << "Failed to get required instance extensions from GLFW";
    return {};
  }
  return std::vector<const char*>(exts, exts + count);
}

// Instance version for IWindow interface - delegates to static
std::vector<const char*> VkWindow::getRequiredInstanceExtensions() const {
  return VkWindow::getRequiredVulkanExtensions();
}

bool VkWindow::create(int width, int height, const std::string& title) {
  if (!glfwInit()) {
    LOG(ERROR) << "Failed to initialize GLFW";
    return false;
  }

  // Don't create OpenGL context, we are using GLFW only for window management
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

  window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
  if (window_ == nullptr) {
    LOG(ERROR) << "Failed to create GLFW window";
    glfwTerminate();
    return false;
  }

  // Store this pointer for callbacks
  glfwSetWindowUserPointer(window_, this);
  glfwSetFramebufferSizeCallback(window_, framebufferResizeCallback);
  glfwSetKeyCallback(window_, keyCallback);
  glfwSetMouseButtonCallback(window_, mouseButtonCallback);
  glfwSetCursorPosCallback(window_, cursorPosCallback);
  glfwSetScrollCallback(window_, scrollCallback);

  LOG(INFO) << "Created window: " << width << "x" << height;
  return true;
}

bool VkWindow::createSurface(VkInstance instance, VkSurfaceKHR* surface) {
  return checkVkResult(
      glfwCreateWindowSurface(instance, window_, nullptr, surface),
      "glfwCreateWindowSurface");
}

void VkWindow::pollEvents() { glfwPollEvents(); }

bool VkWindow::shouldClose() const {
  if (window_ == nullptr) {
    // No window exists - return true to signal exit and prevent null access
    return true;
  }
  // Check if user requested close (X button, Alt+F4, ESC with callback, etc.)
  return glfwWindowShouldClose(window_);
}

void VkWindow::getSize(int* width, int* height) const {
  if (window_ != nullptr) {
    glfwGetWindowSize(window_, width, height);
  } else {
    *width = 0;
    *height = 0;
  }
}

void VkWindow::getFramebufferSize(int* width, int* height) const {
  if (window_ != nullptr) {
    glfwGetFramebufferSize(window_, width, height);
  } else {
    *width = 0;
    *height = 0;
  }
}

bool VkWindow::isMinimized() const {
  if (window_ == nullptr) return true;
  int width, height;
  glfwGetFramebufferSize(window_, &width, &height);
  return width == 0 || height == 0;
}

void VkWindow::resize(int width, int height) {
  if (window_ != nullptr) {
    glfwSetWindowSize(window_, width, height);
  }
}

void VkWindow::setResizeCallback(ResizeCallback callback) {
  resize_callback_ = std::move(callback);
}

void VkWindow::setKeyCallback(KeyCallback callback) {
  key_callback_ = std::move(callback);
}

void VkWindow::setMouseButtonCallback(MouseButtonCallback callback) {
  mouse_button_callback_ = std::move(callback);
}

void VkWindow::setMouseMoveCallback(MouseMoveCallback callback) {
  mouse_move_callback_ = std::move(callback);
}

void VkWindow::setScrollCallback(ScrollCallback callback) {
  scroll_callback_ = std::move(callback);
}

void VkWindow::getCursorPos(double* x, double* y) const {
  if (window_ != nullptr) {
    glfwGetCursorPos(window_, x, y);
  } else {
    *x = 0;
    *y = 0;
  }
}

bool VkWindow::isMouseButtonPressed(int button) const {
  return window_ != nullptr ? glfwGetMouseButton(window_, button) == GLFW_PRESS
                            : false;
}

void VkWindow::framebufferResizeCallback(GLFWwindow* window, int width,
                                         int height) {
  auto* self = static_cast<VkWindow*>(glfwGetWindowUserPointer(window));
  self->framebuffer_resized_ = true;
  if (self->resize_callback_) {
    self->resize_callback_(width, height);
  }
}

void VkWindow::keyCallback(GLFWwindow* window, int key, int /*scancode*/,
                           int action, int mods) {
  auto* self = static_cast<VkWindow*>(glfwGetWindowUserPointer(window));

  // ESC to close window
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
    return;
  }

  if (self->key_callback_) {
    self->key_callback_(key, action, mods);
  }
}

void VkWindow::mouseButtonCallback(GLFWwindow* window, int button, int action,
                                   int mods) {
  auto* self = static_cast<VkWindow*>(glfwGetWindowUserPointer(window));
  if (self->mouse_button_callback_) {
    self->mouse_button_callback_(button, action, mods);
  }
}

void VkWindow::cursorPosCallback(GLFWwindow* window, double x, double y) {
  auto* self = static_cast<VkWindow*>(glfwGetWindowUserPointer(window));
  if (self->mouse_move_callback_) {
    self->mouse_move_callback_(x, y);
  }
}

void VkWindow::scrollCallback(GLFWwindow* window, double x_offset,
                              double y_offset) {
  auto* self = static_cast<VkWindow*>(glfwGetWindowUserPointer(window));
  if (self->scroll_callback_) {
    self->scroll_callback_(x_offset, y_offset);
  }
}

}  // namespace renderer
}  // namespace nvblox
