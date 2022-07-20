# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2022 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

option(CS_PACKAGE_MANAGER "Enable compilation of the package manager" ON)

if (NOT CS_PACKAGE_MANAGER)
  return()
endif()

if (${CMAKE_VERSION} LESS 3.18)
    message("At least CMake 3.18 is required to build the package manager!")
    return()
endif()

# build executable ---------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp)

# Header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES *.hpp)

add_executable(cspm
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

# This property seems to break Ninja on Windows only.
if(NOT (${CMAKE_GENERATOR} STREQUAL "Ninja" AND WIN32))
  # cef needs to be linked first due to https://magpcss.org/ceforum/viewtopic.php?f=6&t=15118&start=10
  # to make sure the link order is kept, we need --no-as-needed
  set_target_properties(cspm PROPERTIES LINK_WHAT_YOU_USE on)
endif()

target_link_libraries(cspm
  cs-utils
)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "package-manager"
  FILES ${SOURCE_FILES} ${HEADER_FILES}
)

# Make sure that CosmoScout VR can be directly started from within Visual Studio.
set_target_properties(cspm PROPERTIES
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}\\bin"
  VS_DEBUGGER_ENVIRONMENT "PATH=..\\lib;%PATH%"
)

# install executable ---------------------------------------------------------------------------------

install(
  TARGETS cspm
  RUNTIME DESTINATION "bin"
)