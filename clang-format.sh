#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                 This file is part of CosmoScout VR                               #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

SRC_DIR="$( cd "$( dirname "$0" )" && pwd )"

find "$SRC_DIR/src" "$SRC_DIR/plugins" -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.inl' \) -exec sh -c '
  for file do
    echo "Formatting $file..."
    clang-format -i "$file"
  done
' sh {} +
