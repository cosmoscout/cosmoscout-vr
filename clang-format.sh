#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                 This file is part of CosmoScout VR                               #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# The src/ and the plugins/ directory is assumed to reside in the same diretory as this script. 
SRC_DIR="$( cd "$( dirname "$0" )" && pwd )"

# Execute clang format for all *.cpp, *.hpp and *.inl files.
find "$SRC_DIR/src" "$SRC_DIR/plugins" "$SRC_DIR/resources" -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.inl' -o -name '*.js' \) -and ! -path '*third-party*' -exec sh -c '
  for file do
    echo "Formatting $file..."
    clang-format -i "$file"
  done
' sh {} +
