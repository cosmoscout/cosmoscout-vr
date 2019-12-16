# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Automatically go through all sub directories. This allows you to add your own plugins
# to CosmoScout VR by putting them into a custom subdirectory here.

FILE(GLOB SUBDIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/*)

foreach(SUBDIR ${SUBDIRECTORIES})
  if(EXISTS "${SUBDIR}/CMakeLists.txt")
    add_subdirectory(${SUBDIR})
  endif()
endforeach()
