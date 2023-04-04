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

---- Original license for this file: ----

 * Copyright (C) 2012-2013 Simon Lynen, ASL, ETH Zurich, Switzerland
 * You can contact the author at <slynen at ethz dot ch>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Adapted from Paul Furgale Schweizer Messer sm_timing*/

#include "nvblox/utils/timing.h"

#include <glog/logging.h>
#include <math.h>
#include <stdio.h>

#include <algorithm>
#include <ostream>
#include <sstream>
#include <string>

namespace nvblox {

namespace timing {

const double kNumSecondsPerNanosecond = 1.e-9;

Timing& Timing::Instance() {
  static Timing t;
  return t;
}

Timing::Timing() : maxTagLength_(0) {}

Timing::~Timing() {}

// Static functions to query the timers:
size_t Timing::GetHandle(std::string const& tag) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  // Search for an existing tag.
  map_t::iterator i = Instance().tagMap_.find(tag);
  if (i == Instance().tagMap_.end()) {
    // If it is not there, create a tag.
    size_t handle = Instance().timers_.size();
    Instance().tagMap_[tag] = handle;
    Instance().timers_.push_back(TimerMapValue());
    // Track the maximum tag length to help printing a table of timing values
    // later.
    Instance().maxTagLength_ = std::max(Instance().maxTagLength_, tag.size());
    return handle;
  } else {
    return i->second;
  }
}

std::string Timing::GetTag(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  std::string tag;

  // Perform a linear search for the tag.
  for (typename map_t::value_type current_tag : Instance().tagMap_) {
    if (current_tag.second == handle) {
      return current_tag.first;
    }
  }
  return tag;
}

// Class functions used for timing.
TimerChrono::TimerChrono(size_t handle, bool constructStopped)
    : timing_(false), handle_(handle) {
  if (!constructStopped) Start();
}

TimerChrono::TimerChrono(std::string const& tag, bool constructStopped)
    : timing_(false), handle_(Timing::GetHandle(tag)) {
  if (!constructStopped) Start();
}

TimerChrono::~TimerChrono() {
  if (IsTiming()) Stop();
}

void TimerChrono::Start() {
  timing_ = true;
  time_ = std::chrono::system_clock::now();
}

void TimerChrono::Stop() {
  std::chrono::time_point<std::chrono::system_clock> now =
      std::chrono::system_clock::now();
  double dt =
      static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(now - time_)
              .count()) *
      kNumSecondsPerNanosecond;

  Timing::Instance().AddTime(handle_, dt);
  timing_ = false;
}

bool TimerChrono::IsTiming() const { return timing_; }

TimerNvtx::TimerNvtx(std::string const& tag, bool constructStopped)
    : timer_(tag, constructStopped), nvtx_range_(tag, constructStopped) {}

TimerNvtx::TimerNvtx(std::string const& tag, const Color& color,
                     bool constructStopped)
    : timer_(tag, constructStopped),
      nvtx_range_(tag, color, constructStopped) {}

void TimerNvtx::Start() {
  timer_.Start();
  nvtx_range_.Start();
}

void TimerNvtx::Stop() {
  timer_.Stop();
  nvtx_range_.Stop();
}

bool TimerNvtx::IsTiming() const { return timer_.IsTiming(); }

void Timing::AddTime(size_t handle, double seconds) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  timers_[handle].acc_.Add(seconds);
}

double Timing::GetTotalSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].acc_.Sum();
}
double Timing::GetTotalSeconds(std::string const& tag) {
  return GetTotalSeconds(GetHandle(tag));
}
double Timing::GetMeanSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].acc_.Mean();
}
double Timing::GetMeanSeconds(std::string const& tag) {
  return GetMeanSeconds(GetHandle(tag));
}
size_t Timing::GetNumSamples(size_t handle) {
  return Instance().timers_[handle].acc_.TotalSamples();
}
size_t Timing::GetNumSamples(std::string const& tag) {
  return GetNumSamples(GetHandle(tag));
}
double Timing::GetVarianceSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].acc_.LazyVariance();
}
double Timing::GetVarianceSeconds(std::string const& tag) {
  return GetVarianceSeconds(GetHandle(tag));
}
double Timing::GetMinSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].acc_.Min();
}
double Timing::GetMinSeconds(std::string const& tag) {
  return GetMinSeconds(GetHandle(tag));
}
double Timing::GetMaxSeconds(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  return Instance().timers_[handle].acc_.Max();
}
double Timing::GetMaxSeconds(std::string const& tag) {
  return GetMaxSeconds(GetHandle(tag));
}

double Timing::GetHz(size_t handle) {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  const double rolling_mean = Instance().timers_[handle].acc_.RollingMean();
  CHECK_GT(rolling_mean, 0.0);
  return 1.0 / rolling_mean;
}

double Timing::GetHz(std::string const& tag) { return GetHz(GetHandle(tag)); }

std::string Timing::SecondsToTimeString(double seconds) {
  char buffer[256];
  snprintf(buffer, sizeof(buffer), "%09.6f", seconds);
  return buffer;
}

void Timing::Print(std::ostream& out) {
  map_t& tagMap = Instance().tagMap_;  // std::map<std::string, size_t>

  if (tagMap.empty()) {
    return;
  }

  out << "NVBlox Timing\n";
  out << "-----------\n";
  out << "operation\tsample_number\ttotal_second\tmean_second\tmin_second\tmax_"
         "second\n";
  out << "-----------\n";
  for (typename map_t::value_type t : tagMap) {
    size_t i = t.second;
    out.width((std::streamsize)Instance().maxTagLength_);
    out.setf(std::ios::left, std::ios::adjustfield);
    out << t.first << "\t";
    out.width(7);

    out.setf(std::ios::right, std::ios::adjustfield);
    out << GetNumSamples(i) << "\t";
    if (GetNumSamples(i) > 0) {
      out << SecondsToTimeString(GetTotalSeconds(i)) << "\t";
      double meansec = GetMeanSeconds(i);
      double stddev = sqrt(GetVarianceSeconds(i));
      out << "(" << SecondsToTimeString(meansec) << " +- ";
      out << SecondsToTimeString(stddev) << ")\t";

      double minsec = GetMinSeconds(i);
      double maxsec = GetMaxSeconds(i);

      // The min or max are out of bounds.
      out << "[" << SecondsToTimeString(minsec) << ","
          << SecondsToTimeString(maxsec) << "]";
    }
    out << std::endl;
  }
}

std::string Timing::Print() {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}

// NOTE(gogojjh): added to only print timeing of key steps (given keywords)
void Timing::Print(std::ostream& out,
                   std::vector<std::string> const& keywords) {
  map_t& tagMap = Instance().tagMap_;  // std::map<std::string, size_t>

  if (tagMap.empty()) {
    return;
  }

  out << "NVBlox Timing\n";
  out << "-----------\n";
  out << "operation\tsample_number\ttotal_second\tmean_second\tmin_second\tmax_"
         "second\n";
  out << "-----------\n";
  for (typename map_t::value_type t : tagMap) {
    bool contain_word = false;
    for (const auto& word : keywords) {
      if (t.first.find(word) != std::string::npos) {
        contain_word = true;
        break;
      }
    }
    if (!contain_word) continue;

    size_t i = t.second;
    out.width((std::streamsize)Instance().maxTagLength_);
    out.setf(std::ios::left, std::ios::adjustfield);
    out << t.first << "\t";
    out.width(7);

    out.setf(std::ios::right, std::ios::adjustfield);
    out << GetNumSamples(i) << "\t";
    if (GetNumSamples(i) > 0) {
      out << SecondsToTimeString(GetTotalSeconds(i)) << "\t";
      double meansec = GetMeanSeconds(i);
      double stddev = sqrt(GetVarianceSeconds(i));
      out << "(" << SecondsToTimeString(meansec) << " +- ";
      out << SecondsToTimeString(stddev) << ")\t";

      double minsec = GetMinSeconds(i);
      double maxsec = GetMaxSeconds(i);

      // The min or max are out of bounds.
      out << "[" << SecondsToTimeString(minsec) << ","
          << SecondsToTimeString(maxsec) << "]";
    }
    out << std::endl;
  }
}

std::string Timing::Print(std::vector<std::string> const& keywords) {
  std::stringstream ss;
  Print(ss, keywords);
  return ss.str();
}

void Timing::Reset() {
  std::lock_guard<std::mutex> lock(Instance().mutex_);
  Instance().tagMap_.clear();
}

}  // namespace timing
}  // namespace nvblox
