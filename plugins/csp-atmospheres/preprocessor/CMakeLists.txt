# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CS_ATMOSPHERE_PREPROCESSOR "Enable compilation of the Atmosphere Preprocessor" OFF)

if (NOT CS_ATMOSPHERE_PREPROCESSOR)
  return()
endif()

# search dependencies ------------------------------------------------------------------------------

find_package(OpenMP REQUIRED)

# build executable ---------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp)

# Header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES *.hpp)

add_executable(atmosphere-preprocessor
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

# This property seems to break Ninja on Windows only.
if(NOT (${CMAKE_GENERATOR} STREQUAL "Ninja" AND WIN32))
  # cef needs to be linked first due to https://magpcss.org/ceforum/viewtopic.php?f=6&t=15118&start=10
  # to make sure the link order is kept, we need --no-as-needed
  set_target_properties(atmosphere-preprocessor PROPERTIES LINK_WHAT_YOU_USE on)
endif()

target_link_libraries(atmosphere-preprocessor
  cs-utils
  OpenMP::OpenMP_CXX
)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "atmosphere-preprocessor"
  FILES ${SOURCE_FILES} ${HEADER_FILES}
)

# Make sure that CosmoScout VR can be directly started from within Visual Studio.
set_target_properties(atmosphere-preprocessor PROPERTIES 
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}\\bin"
  VS_DEBUGGER_ENVIRONMENT "PATH=..\\lib;%PATH%"
)

# install executable ---------------------------------------------------------------------------------

install(
  TARGETS atmosphere-preprocessor
  RUNTIME DESTINATION "bin"
)