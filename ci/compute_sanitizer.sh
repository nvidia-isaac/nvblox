#!/bin/bash

set -exo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

BENCHMARK_CMD=$SCRIPT_DIR/../nvblox/build/tests/benchmark
TEST_DIR=$SCRIPT_DIR/../nvblox/build/tests

# Logging for debug reasons
compute-sanitizer --version
nvidia-smi
ldd $BENCHMARK_CMD
ls /usr/local/cuda/lib64
dpkg -l | grep cuda || true

# Run all sanitizer tools on the benchmark executable
cd $TEST_DIR && compute-sanitizer --error-exitcode=1 --tool memcheck $BENCHMARK_CMD
cd $TEST_DIR && compute-sanitizer --error-exitcode=1 --tool initcheck $BENCHMARK_CMD
cd $TEST_DIR && compute-sanitizer --error-exitcode=1 --tool racecheck $BENCHMARK_CMD
cd $TEST_DIR && compute-sanitizer --error-exitcode=1 --tool synccheck $BENCHMARK_CMD
