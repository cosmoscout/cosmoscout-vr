# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSL_TOOLS "Enable compilation of this plugin" ON)

if (NOT CSL_TOOLS)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)

add_library(csl-tools SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOURCE_FILES}
)

target_link_libraries(csl-tools
  PUBLIC
    cs-core
    cs-utils
)

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csl-tools PROPERTY FOLDER "plugins")

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES}
)


target_include_directories(csl-tools PUBLIC
    ${CMAKE_BINARY_DIR}/plugins/csl-tools
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS csl-tools DESTINATION "lib")

# export header ------------------------------------------------------------------------------------

generate_export_header(csl-tools
    EXPORT_FILE_NAME csl_tools_export.hpp
)
