#!/bin/bash

echo -e "Linting nvblox.h"
echo -e ""

SUCCESS=1

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Get list of files expected in the nvblox.h header.
#These are files which are in include, but not in an internal subfolder.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
HEADER_DIR=$SCRIPT_DIR/../include
HEADER_FILE_LIST=$(find $HEADER_DIR -type f \( -iname "*.h" ! -iname "*nvblox.h" \) ! -path "*/internal/*" -printf '%P\n')

# Check that there are no impl files in public folder locations
for HEADER_FILE in $HEADER_FILE_LIST
do
    if [[ $HEADER_FILE == *"/impl/"* ]]
    then
        echo -e "${RED}Implementation file found in public folder: $HEADER_FILE"
        SUCCESS=0
    fi
done


# Search nvblox.h for each of these files.
NVBLOX_H_PATH=$HEADER_DIR/nvblox/nvblox.h
INCLUDES_STRING=""
AT_LEAST_ONE_HEADER_NOT_FOUND=0
for HEADER_FILE in $HEADER_FILE_LIST
do
    if ! grep -Fq $HEADER_FILE $NVBLOX_H_PATH
    then
        echo -e "${RED}Public header not in nvblox.h: $HEADER_FILE${NC}"
        AT_LEAST_ONE_HEADER_NOT_FOUND=1
    fi
    INCLUDES_STRING+="#include \"$HEADER_FILE\"\n"
done


# If not all headers in, fail and suggest headers to add.
if [ $AT_LEAST_ONE_HEADER_NOT_FOUND == 1 ]
then
    echo -e ""
    echo -e "${RED}Lint failing: Not all public headers are found in nvblox.h${NC}"
    echo -e ""
    echo -e "Replace includes in nvblox.h with the following:"
    echo -e ""
    echo -e $INCLUDES_STRING
    SUCCESS=0
fi

if [ $SUCCESS == 0 ]
then
    echo -e "${RED}Lint of public includes in nvblox.h failed."
    exit 1
else
    echo -e "${GREEN}Lint of public includes in nvblox.h passed."
fi
