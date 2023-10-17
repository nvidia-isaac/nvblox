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
#include "nvblox/utils/nvtx_ranges.h"

#include <functional>

#include <iostream>

namespace nvblox {
namespace timing {

uint32_t colorToUint32(const Color& color) {
  return 0xFF << 24 |                            // NOLINT
         static_cast<uint32_t>(color.r) << 16 |  // NOLINT
         static_cast<uint32_t>(color.g) << 8 |   // NOLINT
         static_cast<uint32_t>(color.b);         // NOLINT
}

const uint32_t nxtx_ranges_colors[] = {
    colorToUint32(Color::Black()),  colorToUint32(Color::Gray()),
    colorToUint32(Color::Red()),    colorToUint32(Color::Green()),
    colorToUint32(Color::Blue()),   colorToUint32(Color::Yellow()),
    colorToUint32(Color::Orange()), colorToUint32(Color::Purple()),
    colorToUint32(Color::Teal()),   colorToUint32(Color::Pink())};
const int num_colors = sizeof(nxtx_ranges_colors) / sizeof(uint32_t);

uint32_t colorFromString(const std::string& str) {
  return nxtx_ranges_colors[std::hash<std::string>{}(str) % num_colors];
}

NvtxRange::NvtxRange(const std::string& message, const Color& color,
                     bool construct_stopped)
    : started_(false), event_attributes_{} {
  Init(message, color);
  if (!construct_stopped) Start();
}

NvtxRange::NvtxRange(const std::string& message, bool construct_stopped)
    : started_(false), event_attributes_{} {
  Init(message, colorFromString(message));
  if (!construct_stopped) Start();
}

NvtxRange::~NvtxRange() { Stop(); }

void NvtxRange::Start() {
  started_ = true;
  id_ = nvtxRangeStartEx(&event_attributes_);
}

void NvtxRange::Stop() {
  if (started_) nvtxRangeEnd(id_);
  started_ = false;
}

void NvtxRange::Init(const std::string& message, const Color& color) {
  Init(message, colorToUint32(color));
}

void NvtxRange::Init(const std::string& message, const uint32_t color) {
  // Initialize
  event_attributes_ = nvtxEventAttributes_t{};
  event_attributes_.version = NVTX_VERSION;
  event_attributes_.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  // Configure the Attributes
  event_attributes_.colorType = NVTX_COLOR_ARGB;
  event_attributes_.color = color;
  event_attributes_.messageType = NVTX_MESSAGE_TYPE_ASCII;
  event_attributes_.message.ascii = message.c_str();
}

void mark(const std::string& message, const uint32_t color) {
  // Initialize
  nvtxEventAttributes_t event_attributes{};
  event_attributes.version = NVTX_VERSION;
  event_attributes.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  event_attributes.colorType = NVTX_COLOR_ARGB;
  event_attributes.color = color;
  event_attributes.messageType = NVTX_MESSAGE_TYPE_ASCII;
  event_attributes.message.ascii = message.c_str();
  nvtxMarkEx(&event_attributes);
}

void mark(const std::string& message) {
  mark(message, colorFromString(message));
}

void mark(const std::string& message, const Color& color) {
  mark(message, colorToUint32(color));
}

}  // namespace timing
}  // namespace nvblox
