# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# Automatically go through all sub directories. This allows you to add your own plugins
# to CosmoScout VR by putting them into a custom subdirectory here.

FILE(GLOB SUBDIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(SUBDIR ${SUBDIRECTORIES})
  if(EXISTS "${SUBDIR}/CMakeLists.txt")
    add_subdirectory(${SUBDIR})
  endif()
endforeach()
