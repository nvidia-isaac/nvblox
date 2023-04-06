#pragma once

#include "nvblox/core/unified_ptr.h"
#include "nvblox/integrators/weighting_function.h"

namespace nvblox {
namespace test_utils {

float computeWeight(unified_ptr<WeightingFunction>& weighting_function,
                    float point_distance_from_camera,
                    float voxel_distance_from_camera,
                    float truncation_distance);

unified_ptr<WeightingFunction> createWeightingFunction(
    WeightingFunctionType type) {
  return make_unified<WeightingFunction>(type);
}

}  // namespace test_utils
}  // namespace nvblox
