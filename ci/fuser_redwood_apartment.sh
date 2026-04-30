#!/bin/bash

# Stability test that runs through the redwood apartment dataset (~30k frames).
set -exo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
EXEC_DIR=$SCRIPT_DIR/../build/executables

# Download the test dataset
wget https://urm.nvidia.com/artifactory/sw-isaac-sdk-generic-local/dependencies/internal/data/redwood_apartment.tar
tar -xvf redwood_apartment.tar > /dev/null

# Run the thing
$EXEC_DIR/fuse_redwood redwood/apartment
