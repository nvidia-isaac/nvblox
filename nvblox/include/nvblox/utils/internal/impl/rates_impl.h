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

namespace nvblox {
namespace timing {

template <typename T, int kLength>
void CircularBuffer<T, kLength>::push(const T& el) {
  buffer_[head_] = el;
  if (full_) {
    tail_ = (tail_ + 1) % kLength;
  }
  head_ = (head_ + 1) % kLength;
  full_ = head_ == tail_;
}

template <typename T, int kLength>
T CircularBuffer<T, kLength>::oldest() const {
  if (empty()) {
    return T();
  }
  return buffer_[tail_];
}

template <typename T, int kLength>
T CircularBuffer<T, kLength>::newest() const {
  if (empty()) {
    return T();
  }
  const int newest_idx = (head_ == 0) ? (kLength - 1) : (head_ - 1);
  return buffer_[newest_idx];
}

template <typename T, int kLength>
bool CircularBuffer<T, kLength>::full() const {
  return full_;
}

template <typename T, int kLength>
bool CircularBuffer<T, kLength>::empty() const {
  return (!full_ && (head_ == tail_));
}

template <typename T, int kLength>
int CircularBuffer<T, kLength>::size() const {
  if (full_) {
    return kLength;
  }
  if (head_ >= tail_) {
    return head_ - tail_;
  } else {
    return kLength + head_ - tail_;
  }
}

template <typename T, int kLength>
void CircularBuffer<T, kLength>::reset() {
  head_ = 0;
  tail_ = 0;
  full_ = false;
}

}  // namespace timing
}  // namespace nvblox
