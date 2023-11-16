/*
Copyright 2023 NVIDIA CORPORATION

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

#include <cuda_runtime.h>

namespace nvblox {

/// Type to store time.
/// This is strongly typed to int64_t to avoid potentially dangerous implicit
/// conversions.
class Time {
 public:
  __host__ __device__ explicit Time(int64_t time) : time_(time) {}
  // Conversion operator to int64_t
  explicit operator int64_t() const { return time_; }

  // Heaps of (trivial) operator overloading
  __host__ __device__ bool operator==(const Time& other) const {
    return time_ == other.time_;
  }
  __host__ __device__ bool operator<(const Time& other) const {
    return time_ < other.time_;
  }
  __host__ __device__ bool operator>(const Time& other) const {
    return time_ > other.time_;
  }
  __host__ __device__ bool operator<=(const Time& other) const {
    return !(time_ > other.time_);
  }
  __host__ __device__ bool operator>=(const Time& other) const {
    return !(time_ < other.time_);
  }
  __host__ __device__ Time operator+=(const Time& other) {
    time_ += other.time_;
    return *this;
  }
  __host__ __device__ Time operator-=(const Time& other) {
    time_ -= other.time_;
    return *this;
  }
  __host__ __device__ Time operator+(const Time& other) const {
    return Time(time_ + other.time_);
  }
  __host__ __device__ Time operator-(const Time& other) const {
    return Time(time_ - other.time_);
  }

 private:
  // The actual time data
  int64_t time_;
};

__host__ inline Time operator*(int scalar, Time time) {
  return Time(scalar * static_cast<int64_t>(time));
}

}  // namespace nvblox
