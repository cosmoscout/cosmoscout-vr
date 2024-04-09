# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSP_ATMOSPHERES_PREPROCESSOR "Enable compilation of the Atmosphere Preprocessor" OFF)

if (NOT CSP_ATMOSPHERES_PREPROCESSOR)
  return()
endif()

# build executable ---------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp)

# Header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES *.hpp)

add_executable(bruneton-preprocessor
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

target_link_libraries(bruneton-preprocessor
  cs-core
)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "bruneton-preprocessor"
  FILES ${SOURCE_FILES} ${HEADER_FILES}
)

# Make sure that the tool can be directly started from within Visual Studio.
set_target_properties(bruneton-preprocessor PROPERTIES 
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}\\bin"
  VS_DEBUGGER_ENVIRONMENT "PATH=..\\lib;%PATH%"
)

# install executable ---------------------------------------------------------------------------------

install(
  TARGETS bruneton-preprocessor
  RUNTIME DESTINATION "bin"
)