# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CS_ATMOSPHERE_GENERATOR "Enable compilation of the Atmosphere Generator" OFF)

if (NOT CS_ATMOSPHERE_GENERATOR)
  return()
endif()

# build executable ---------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp)

# Header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES *.hpp)

add_executable(atmosphere-generator
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

# This property seems to break Ninja on Windows only.
if(NOT (${CMAKE_GENERATOR} STREQUAL "Ninja" AND WIN32))
  # cef needs to be linked first due to https://magpcss.org/ceforum/viewtopic.php?f=6&t=15118&start=10
  # to make sure the link order is kept, we need --no-as-needed
  set_target_properties(atmosphere-generator PROPERTIES LINK_WHAT_YOU_USE on)
endif()

target_link_libraries(atmosphere-generator
  cs-utils
)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "atmosphere-generator"
  FILES ${SOURCE_FILES} ${HEADER_FILES}
)

# Make sure that CosmoScout VR can be directly started from within Visual Studio.
set_target_properties(atmosphere-generator PROPERTIES 
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}\\bin"
  VS_DEBUGGER_ENVIRONMENT "PATH=..\\lib;%PATH%"
)

# install executable ---------------------------------------------------------------------------------

install(
  TARGETS atmosphere-generator
  RUNTIME DESTINATION "bin"
)