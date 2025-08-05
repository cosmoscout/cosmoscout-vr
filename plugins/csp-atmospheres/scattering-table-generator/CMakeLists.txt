# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSP_ATMOSPHERES_SCATTERING_TABLE_GENERATOR "Enable compilation of the Atmosphere CSV Generator" OFF)

if (NOT CSP_ATMOSPHERES_SCATTERING_TABLE_GENERATOR)
  return()
endif()

# search dependencies ------------------------------------------------------------------------------

find_package(OpenMP REQUIRED)

# build executable ---------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp)

# Header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES *.hpp)

add_executable(scattering-table-generator
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

target_link_libraries(scattering-table-generator
  cs-utils
  OpenMP::OpenMP_CXX
)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "scattering-table-generator"
  FILES ${SOURCE_FILES} ${HEADER_FILES}
)

# Make sure that CosmoScout VR can be directly started from within Visual Studio.
set_target_properties(scattering-table-generator PROPERTIES 
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}\\bin"
  VS_DEBUGGER_ENVIRONMENT "PATH=..\\lib;%PATH%"
)

# install executable ---------------------------------------------------------------------------------

install(
  TARGETS scattering-table-generator
  RUNTIME DESTINATION "bin"
)