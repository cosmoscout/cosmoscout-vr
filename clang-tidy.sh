#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                 This file is part of CosmoScout VR                               #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# The install/, build/, src/ and the plugins/ directories is assumed to reside in the same diretory
# as this script. 
SCRIPT_DIR="$( cd "$( dirname "$0" )" && pwd )"

FLAGS="-std=c++17 "
FLAGS+="-DGLM_ENABLE_EXPERIMENTAL "
FLAGS+="-DGLM_FORCE_SWIZZLE "
FLAGS+="-I${SCRIPT_DIR}/src "
FLAGS+="-I${SCRIPT_DIR}/install/linux-externals-release/include "
FLAGS+="-I${SCRIPT_DIR}/build/linux-release/src/cs-utils "
FLAGS+="-I${SCRIPT_DIR}/build/linux-release/src/cs-scene "
FLAGS+="-I${SCRIPT_DIR}/build/linux-release/src/cs-graphics "
FLAGS+="-I${SCRIPT_DIR}/build/linux-release/src/cs-core "
FLAGS+="-I${SCRIPT_DIR}/build/linux-release/src/cs-gui "

CHECKS="modernize-*,"
CHECKS+="bugprone-*,"
CHECKS+="google-*,"
CHECKS+="readability-*,"
CHECKS+="performance-*,"
CHECKS+="hicpp-*,"
CHECKS+="misc-*,"

# run clang-tidy on all source files
find src -iname "*.cpp"|while read file; do
  echo "Tidying ${file}..."
  clang-tidy-7 -checks=$CHECKS -fix -quiet $file -- $FLAGS
done
