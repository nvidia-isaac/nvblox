/*
Copyright 2025 NVIDIA CORPORATION

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

// Stub implementation of FuserVisualizer used when BUILD_RENDERER=OFF.
// All methods are no-ops

#include "nvblox/fuser/fuser_visualizer.h"

#include <glog/logging.h>

namespace nvblox {

struct FuserVisualizer::Impl {};

FuserVisualizer::FuserVisualizer() : impl_(std::make_unique<Impl>()) {}

FuserVisualizer::~FuserVisualizer() = default;

bool FuserVisualizer::init(const std::string& /*title*/,
                           std::shared_ptr<CudaStreamOwning> /*stream*/) {
  LOG(WARNING)
      << "Visualization disabled: nvblox was built without BUILD_RENDERER. "
         "Re-build with -DBUILD_RENDERER=ON to enable.";
  return false;
}

void FuserVisualizer::updateMesh(const ColorMesh& /*mesh*/,
                                 const Camera& /*color_cam*/,
                                 const Transform& /*T_C_L*/,
                                 const ColorImage& /*color_frame*/,
                                 const DepthImage& /*depth_frame*/) {}

bool FuserVisualizer::renderAndPoll() { return true; }

bool FuserVisualizer::isPaused() const { return false; }

bool FuserVisualizer::isAvailable() const { return false; }

}  // namespace nvblox
