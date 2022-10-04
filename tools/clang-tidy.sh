#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# For this script to work, CosmoScout has to be built at least once in release mode, as the
# build/linux-Release directory is required. The directory is assumed to reside next to this script
# or one level above.
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Run clang-tidy on all source files
if [ -d "$SCRIPT_DIR/../build/linux-Release" ]
then
  run-clang-tidy -fix -quiet -p "$SCRIPT_DIR/../build/linux-Release"
elif [ -d "$SCRIPT_DIR/../../build/linux-Release" ]
then
  run-clang-tidy -fix -quiet -p "$SCRIPT_DIR/../../build/linux-Release"
else
    echo "Failed to find build directory!"
fi

