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

/// @file error_check.h
/// @brief Vulkan error checking utilities.
///
/// Two error handling approaches:
/// - checkVkErrors(): Fatal (LOG(FATAL)). Use for unrecoverable errors.
/// - checkVkResult(): Returns false. Use for recoverable errors
///   (e.g., swapchain resize).

#include <cstdlib>

#include <glog/logging.h>
#include <vulkan/vulkan.h>

namespace nvblox {
namespace renderer {

/// Soft check: returns false on failure, logs error. Use for recoverable errors
/// (e.g., swapchain out of date).
/// @param result The VkResult to check.
/// @param operation Description of the operation for error logging.
/// @return true if result == VK_SUCCESS, false otherwise.
inline bool checkVkResult(VkResult result, const char* operation) {
  if (result != VK_SUCCESS) {
    LOG(ERROR) << operation << " failed with VkResult: " << result;
    return false;
  }
  return true;
}

/// Warning-level check for destructors: never throws/aborts, only logs warning.
/// Use in destructors where we cannot propagate errors but want to log issues.
/// @param result The VkResult to check.
/// @param operation Description of the operation for warning logging.
/// @return true if result == VK_SUCCESS, false otherwise.
inline bool checkVkResultWarn(VkResult result, const char* operation) {
  if (result != VK_SUCCESS) {
    LOG(WARNING) << operation << " failed with VkResult: " << result;
    return false;
  }
  return true;
}

/// Hard check: aborts on failure (nvblox style). Use for fatal errors.
/// @param result The VkResult to check.
/// @param func The function/expression string.
/// @param file Source file name.
/// @param line Source line number.
inline void check_vk_error_value(VkResult result, const char* func,
                                 const char* file, int line) {
  if (result != VK_SUCCESS) {
    LOG(FATAL) << file << ":" << line << " Vulkan error in " << func
               << " failed with VkResult: " << result;
    std::abort();
  }
}

/// Macro for fatal Vulkan error checking (matches nvblox checkCudaErrors style)
#define checkVkErrors(val) \
  nvblox::renderer::check_vk_error_value((val), #val, __FILE__, __LINE__)

}  // namespace renderer
}  // namespace nvblox
