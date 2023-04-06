#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
FILES_DIR=$SCRIPT_DIR/..

FILES=$( find $FILES_DIR -type f \( -iname "*.h" -o -iname "*.cpp"  -o -iname "*.cuh" -o -iname "*.cu" \) ! -path "*/build/*" ! -path "*/install/*" ! -path "*/thirdparty/*" ! -path "*/external/*" ) #|xargs -I {} clang-format -i {}

clang-format -i $FILES
