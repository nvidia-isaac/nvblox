#!/usr/bin/env bash
# Copyright 2023 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script allow us to test that we do not accidentally introduce
# more work on the default cuda stream. Using the default stream is
# undesireable since it will synchronize the whole device.  The script
# measures the number of API calls placed on the cuda stream as a
# fraction of the total number of calls.


# Make the shell script more robust
set -euxo pipefail

# Make sure that the expected version of nsys is used
NSYS=/usr/local/cuda-11.8/bin/nsys

# Command to benchmark. We run a fixed number of iterations for to make it reproducable
BENCHMARK_CMD="../executables/benchmark --benchmark_filter=benchmarkAll/iterations:100"

# Name of nsys profile output files
REPORT_NAME="profile_report"

# Max allowed percentage of calls using default stream. Do NOT
# increase this number unless there is a very good reason for doing so.
THRESHOLD=0

# Helper function for running nsys and export to a txt file.
function run_nsys ()
{
    rm -f "$REPORT_NAME".txt "$REPORT_NAME".nsys-rep
    $NSYS profile --force-overwrite true --trace=cuda --output "$REPORT_NAME" $1
    $NSYS export "$REPORT_NAME".nsys-rep --type text -o "$REPORT_NAME".txt
}

# Obtain ID of default stream as it may differ between systems. This
# is done by running an executable that only performs work on the
# default stream and parsing its nsys profiling output.
run_nsys run_memcpy_on_default_cuda_stream
DEFAULT_STREAM_ID=$(cat "$REPORT_NAME".txt  | grep streamId | cut --delimiter=: --field=2 | tr --delete ' ')

# Run benchmark with nsys
run_nsys "$BENCHMARK_CMD"

# Count total number of stream events
NUM_TOTAL_EVENTS=$(cat "$REPORT_NAME".txt  | grep "streamId:" | wc -l)

if (( $NUM_TOTAL_EVENTS < 10000 ))
then
    echo "Too few stream events recorded"
    exit 1
fi

# Count number of events on the default stream
NUM_DEFAULT_EVENTS=$(cat "$REPORT_NAME".txt  | egrep "streamId: $DEFAULT_STREAM_ID$" | wc -l)

# Compute the percentage of default events
PERCENT_DEFAULT_EVENTS=$(python3 -c "print(int(100 * $NUM_DEFAULT_EVENTS / $NUM_TOTAL_EVENTS))")

set +x
echo "${PERCENT_DEFAULT_EVENTS}% of all CUDA calls use the default stream"
echo "Threshold: ${THRESHOLD}%"

if (( $PERCENT_DEFAULT_EVENTS > $THRESHOLD ))
then
    echo "ERROR: Threshold exeeded"
    exit 1
else
    echo "PASSED"
    exit 0
fi


