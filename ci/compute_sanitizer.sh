#!/bin/bash

# Run selected tests under cuda's compute-sanitizer
# TODO(dtingdahl) integrate with ctest

set -exo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

TEST_DIR=$SCRIPT_DIR/../build/nvblox/tests
BENCHMARK_DIR=$SCRIPT_DIR/../build/executables

# Logging for debug reasons
compute-sanitizer --version
nvidia-smi
ls /usr/local/cuda/lib64
dpkg -l | grep cuda || true

# List of tests to run under compute-sanitizer
# Add tests that are cuda-intensive and lightweight
TESTS_TO_RUN=("$BENCHMARK_DIR/nvblox_benchmark --benchmark_filter=benchmarkAll"
              "test_color_integrator --gtest_filter=ColorIntegrationTest.IntegrateColorToGroundTruthDistanceField"
              "test_esdf_integrator --gtest_filter=ParameterizedEsdfTests/EsdfIntegratorTest.OccupancySingleEsdfTestGPU/0"
              "test_freespace_integrator"
              "test_gpu_hash_interface"
              "test_gpu_layer_view"
              "test_layer_serializer_gpu"
              "test_lidar_integration"
              "test_mesh_serializer"
              "test_occupancy_integrator --gtest_filter=*ReconstructPlane*"
              "test_occupancy_decay"
              "test_tsdf_integrator --gtest_filter=*ReconstructPlane*"
              "test_tsdf_decay"
             )

# Create a txt file with all commands to run
JOB_FILE=$(mktemp)
for test_cmd in "${TESTS_TO_RUN[@]}"
do
    echo "compute-sanitizer --error-exitcode=1 --tool memcheck $test_cmd" >> $JOB_FILE
    echo "compute-sanitizer --error-exitcode=1 --tool initcheck $test_cmd" >> $JOB_FILE
    echo "compute-sanitizer --error-exitcode=1 --tool racecheck $test_cmd" >> $JOB_FILE
    echo "compute-sanitizer --error-exitcode=1 --tool synccheck $test_cmd" >> $JOB_FILE
done

# Launch all commands in parallel
(
    cd $TEST_DIR
    # Allow only 4 simultaneous jobs to reduce GPU memory usage.
    cat $JOB_FILE | parallel --halt-on-error now,fail=1  -j4
    rm $JOB_FILE
)
