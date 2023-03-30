/*
Copyright 2022 NVIDIA CORPORATION

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

#include <string>

#include <nvToolsExt.h>

#include "nvblox/core/color.h"

namespace nvblox {
namespace timing {

/// Instrument our timers with NvtxRanges, which can be visualized in Nsight
/// Systems to aid with debugging and profiling.
class NvtxRange {
 public:
  NvtxRange(const std::string& message, const Color& color,
            bool constructed_stopped = false);
  NvtxRange(const std::string& message, bool constructed_stopped = false);
  ~NvtxRange();

  NvtxRange() = delete;
  NvtxRange(const NvtxRange& other) = delete;

  void Start();
  void Stop();

  bool Started() const { return started_; };

 private:
  void Init(const std::string& message, const Color& color);
  void Init(const std::string& message, const uint32_t color);

  bool started_;
  nvtxEventAttributes_t event_attributes_;
  nvtxRangeId_t id_;
};

void mark(const std::string& message, const Color& color);
void mark(const std::string& message);

}  // namespace timing
}  // namespace nvblox
