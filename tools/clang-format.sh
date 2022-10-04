#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# The src/, resources/, tools/ and the plugins/ directory is assumed to reside one directory above
# this script.
SRC_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Execute clang format for all *.cpp, *.hpp and *.inl files.
find "$SRC_DIR/../src" "$SRC_DIR/../plugins" "$SRC_DIR/../resources" "$SRC_DIR/../tools" -type f \
     \( -name '*.cpp' -o -name '*.hpp' -o -name '*.inl' -o \
        -name '*.cu' -o -name '*.cuh' -o -name '*.js' \) -and ! -path '*third-party*' -exec sh -c '
  for file do
    echo "Formatting $file..."
    clang-format -i "$file"
  done
' sh {} +
