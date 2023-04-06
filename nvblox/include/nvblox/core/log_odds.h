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

#include <cuda_runtime.h>
#include <cmath>

namespace nvblox {

__host__ __device__ float inline logOddsFromProbability(float probability) {
  // make sure log odds is bounded
  constexpr float min_probability = 1e-3f;
  constexpr float max_probability = 1.0f - 1e-3f;
  probability = fmax(min_probability, fmin(probability, max_probability));

  return log(probability / (1.0f - probability));
}

__host__ __device__ float inline probabilityFromLogOdds(float log_odds) {
  return exp(log_odds) / (1.0f + exp(log_odds));
}

}  // namespace nvblox
