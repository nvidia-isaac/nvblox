#include "nvblox/utils/rates.h"

#include <iostream>

namespace nvblox {
namespace timing {

void Ticker::tick(int64_t timestamp_ns) { circular_buffer_.push(timestamp_ns); }

float Ticker::getMeanRateHz() const {
  if (circular_buffer_.empty()) {
    return 0.0f;
  }
  const int64_t time_span_ns =
      circular_buffer_.newest() - circular_buffer_.oldest();
  if (time_span_ns <= 0) {
    return 0.0f;
  }
  constexpr int64_t kSecondsToNanoSeconds = 1e9;
  return static_cast<float>(
      static_cast<double>(circular_buffer_.size() * kSecondsToNanoSeconds) /
      static_cast<double>(time_span_ns));
}

int Ticker::getNumSamples() const { return circular_buffer_.size(); }

void ChronoTicker::tick() {
  const std::chrono::time_point<std::chrono::system_clock> now =
      std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  auto timestamp_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
  this->Ticker::tick(timestamp_ns.count());
}

Rates& Rates::getInstance() {
  static Rates rates;
  return rates;
}

ChronoTicker& Rates::getTicker(const std::string& tag) {
  std::lock_guard<std::mutex> lock(getInstance().mutex_);
  TickerMap& tickers = getInstance().tickers_;
  auto it = tickers.find(tag);
  if (it != tickers.end()) {
    // Ticker already exists so return what we've found.
    return it->second;
  } else {
    // This tag hasn't been ticked before. Let's create it, and return the new
    // ticker.
    auto insert_status = tickers.emplace(tag, ChronoTicker());
    getInstance().max_tag_length_ =
        std::max(getInstance().max_tag_length_, tag.size());
    return insert_status.first->second;
  }
}

void Rates::tick(const std::string& tag) {
  ChronoTicker& ticker = getInstance().getTicker(tag);
  std::lock_guard<std::mutex> lock(getInstance().mutex_);
  ticker.tick();
}

float Rates::getMeanRateHz(const std::string& tag) {
  if (!getInstance().exists(tag)) {
    return 0.0f;
  }
  const ChronoTicker& ticker = getInstance().getTicker(tag);
  std::lock_guard<std::mutex> lock(getInstance().mutex_);
  return ticker.getMeanRateHz();
}

std::vector<std::string> Rates::getTags() {
  const Rates& rates = Rates::getInstance();
  std::vector<std::string> keys;
  keys.reserve(rates.tickers_.size());
  for (const auto& [tag, value] : rates.tickers_) {
    keys.push_back(tag);
  }
  return keys;
}

std::string Rates::rateToString(float rate_hz) {
  char buffer[256];
  snprintf(buffer, sizeof(buffer), "%0.1f", rate_hz);
  return buffer;
}

void Rates::Print(std::ostream& out) {
  out << "\nNVBlox Rates (in Hz)\n";
  out << "namespace/tag - NumSamples (Window Length) - Mean \n";
  out << "-----------\n";

  for (const auto& tag_ticker_pair : getInstance().tickers_) {
    const std::string& ticker_name = tag_ticker_pair.first;

    out.width(static_cast<std::streamsize>(getInstance().max_tag_length_));
    out.setf(std::ios::left, std::ios::adjustfield);
    out << ticker_name << "\t";
    out.width(7);

    const ChronoTicker& ticker = tag_ticker_pair.second;
    const int num_samples = ticker.getNumSamples();
    out << num_samples << "\t";
    if (num_samples > 0) {
      const float mean_rate = ticker.getMeanRateHz();
      out << rateToString(mean_rate);
    }
    out << std::endl;
  }
  out << "-----------\n";
}

std::string Rates::Print() {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}

}  // namespace timing
}  // namespace nvblox
