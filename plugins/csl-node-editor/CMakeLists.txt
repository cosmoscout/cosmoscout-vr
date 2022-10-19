# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSL_NODE_EDITOR "Enable compilation of this plugin" ON)

if (NOT CSL_NODE_EDITOR)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)

add_library(csl-node-editor SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

target_link_libraries(csl-node-editor
  PUBLIC
    cs-core
    cs-utils
    civetweb::civetweb
    civetweb::civetwebcpp
)

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csl-node-editor PROPERTY FOLDER "plugins")

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES}
)


target_include_directories(csl-node-editor PUBLIC
    ${CMAKE_BINARY_DIR}/plugins/csl-node-editor
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS csl-node-editor DESTINATION "lib")

# export header ------------------------------------------------------------------------------------

generate_export_header(csl-node-editor
    EXPORT_FILE_NAME csl_node_editor_export.hpp
)
