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

#include <string>
#include <unordered_map>

#include <glog/logging.h>

#include <nvblox/utils/timing.h>

namespace nvblox {
namespace timing {

template <typename T, int kLength>
class CircularBuffer {
 public:
  CircularBuffer() = default;
  virtual ~CircularBuffer() = default;

  /// Pushes a new element into the buffer, deleting the oldest if the buffer is
  /// full.
  /// @param el The element to be added.
  void push(const T& el);

  /// Get the oldest element in the buffer.
  /// @return A copy of the oldest element
  T oldest() const;

  /// Get the newest element in the buffer.
  /// @return A copy of the newest element
  T newest() const;

  /// Return a flag indicating if the buffer full (i.e. is it overwritting old
  /// elements when new data is added).
  /// @return True if the buffer is full
  bool full() const;

  /// Return a flag indicating if the buffer has nothing in it.
  /// @return True if the buffer is empty
  bool empty() const;

  /// How many samples in the buffer. Maxes out at kLength.
  /// @return the number of samples in the buffer.
  int size() const;

  /// Clear the buffer
  void reset();

 protected:
  std::array<T, kLength> buffer_;
  int head_ = 0;
  int tail_ = 0;
  bool full_ = false;
};

class Ticker {
 public:
  Ticker() = default;
  virtual ~Ticker() = default;

  /// Adds a timestamp to the window of timestamps used to calculate the average
  /// rate. The window is a circular buffer so new measurements kick out old
  /// measurements once the window is full.
  /// @param timestamp_ns The timestamp in nanoseconds
  void tick(int64_t timestamp_ns);

  /// Gets the average rate that the ticker was ticked over the window.
  /// @return The mean tick-rate in Hz
  float getMeanRateHz() const;

  /// The number of samples in the buffer. Obviously the maximum number of
  /// samples is the buffer length.
  /// @return The number of samples in the circular buffer used to calculate the
  /// rate.
  int getNumSamples() const;

 protected:
  static constexpr int kBufferLength = 100;
  CircularBuffer<int64_t, kBufferLength> circular_buffer_;
};

/// A class named after the greatest RPG of all time. Also, a rate ticker that
/// uses the system clock through std::chrono.
class ChronoTicker : public Ticker {
 public:
  ChronoTicker() = default;
  virtual ~ChronoTicker() = default;

  /// Add a timestamp sample to the buffer used for rate estimation.
  void tick();
};

class Rates {
 public:
  /// Main input interface. Add a rate measurement to a tag. Internally the
  /// singleton uses the time between this call and the last call to calculate
  /// the rate and adds it to a running average measurement.
  /// @param tag A unique string identifying this rate-measurer.
  static void tick(const std::string& tag);

  /// Output interface. Prints a table of the rates of the ticked tags.
  /// @param out The stream to be printed to.
  static void Print(std::ostream& out);

  /// Output interface. Prints a table of the rates of the ticked tags.
  /// @return A table of the rates of the ticked tags as a string.
  static std::string Print();

  // Get a reference to the Rates singleton
  /// @return A reference to this program's rates singleton
  static Rates& getInstance();

  /// Returns a reference to a ticker, if it exists. If the ticker does not
  /// already exist we create a new ticker with this tag and return it.
  /// @param tag A unique string identifying this rates-measurer
  /// @return A reference to the ticker doing the rate calculation.
  static ChronoTicker& getTicker(const std::string& tag);

  /// Gets the mean rate current calculated by one of the rate-measurers
  /// @param tag The tag of the rate-measurer.
  /// @return The rate. Returns 0.0 Hz if the ticker doesn't exist.
  static float getMeanRateHz(const std::string& tag);

  /// Gets a list of all rate tags that have been registered.
  /// @return The list of tags.
  static std::vector<std::string> getTags();

  static bool exists(const std::string& tag) {
    return !(getInstance().tickers_.find(tag) == getInstance().tickers_.end());
  }

 protected:
  Rates() = default;
  ~Rates() = default;

  // Formats a floating point rate for writing to our table string.
  static std::string rateToString(float rate_hz);

  using TickerMap = std::unordered_map<std::string, ChronoTicker>;

  std::mutex mutex_;
  TickerMap tickers_;
  size_t max_tag_length_ = 0;
};

}  // namespace timing
}  // namespace nvblox

#include "nvblox/utils/internal/impl/rates_impl.h"
