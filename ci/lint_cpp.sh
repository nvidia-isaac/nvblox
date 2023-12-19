#!/bin/bash

set -exuo pipefail

# Passing RUNNING_IN_CI as argument cause this script to run in CI-mode.
# Othwerwise local mode will be used
#
# CI-mode:     Any violation of the formatting rule will trigger error
# local mode:  Violations of formatting rules are auto corrected in-place
RUNNING_IN_CI=0
[[ ($# == 1) && ($1 == "RUNNING_IN_CI") ]] && RUNNING_IN_CI=1

# Find CPP files
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
FILES_DIR=$SCRIPT_DIR/..
CPP_FILES=$( find $FILES_DIR -type f \( -iname "*.h" -o -iname "*.cpp"  -o -iname "*.cuh" -o -iname "*.cu" \) ! -path "*/build/*" ! -path "*/install/*" ! -path "*/thirdparty/*" ! -path "*/external/*" )

# Apply Clang format
CLANG_FORMAT_ARGS=""
[[ $RUNNING_IN_CI == 1 ]] && CLANG_FORMAT_ARGS="-Werror --dry-run"
clang-format $CLANG_FORMAT_ARGS -i $CPP_FILES
