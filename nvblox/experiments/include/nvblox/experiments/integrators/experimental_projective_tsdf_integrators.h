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

#include "nvblox/integrators/projective_tsdf_integrator.h"

namespace nvblox {
namespace experiments {

class ProjectiveTsdfIntegratorExperimentsBase
    : public ProjectiveTsdfIntegrator {
 public:
  ProjectiveTsdfIntegratorExperimentsBase();
  virtual ~ProjectiveTsdfIntegratorExperimentsBase();

  void finish() const;

 protected:
};

class ProjectiveTsdfIntegratorExperimentsTexture
    : public ProjectiveTsdfIntegratorExperimentsBase {
 public:
  ProjectiveTsdfIntegratorExperimentsTexture()
      : ProjectiveTsdfIntegratorExperimentsBase(){};
  virtual ~ProjectiveTsdfIntegratorExperimentsTexture(){};

 protected:
  void updateBlocks(const std::vector<Index3D>& block_indices,
                    const DepthImage& depth_frame, const Transform& T_L_C,
                    const Camera& camera, const float truncation_distance_m,
                    TsdfLayer* layer);
};

class ProjectiveTsdfIntegratorExperimentsGlobal
    : public ProjectiveTsdfIntegratorExperimentsBase {
 public:
  ProjectiveTsdfIntegratorExperimentsGlobal()
      : ProjectiveTsdfIntegratorExperimentsBase(){};
  virtual ~ProjectiveTsdfIntegratorExperimentsGlobal(){};

 protected:
  void updateBlocks(const std::vector<Index3D>& block_indices,
                    const DepthImage& depth_frame, const Transform& T_L_C,
                    const Camera& camera, const float truncation_distance_m,
                    TsdfLayer* layer);
};

}  //  namespace experiments
}  //  namespace nvblox
