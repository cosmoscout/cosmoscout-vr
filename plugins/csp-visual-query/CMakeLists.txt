# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSP_VISUAL_QUERY "Enable compilation of this plugin" ON)

if (NOT CSP_VISUAL_QUERY)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp src/*/*.cpp)
file(GLOB NODE_FILES src/*/*.js)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp src/*/*.cpp)
file(GLOB_RECURSE RESOURCE_FILES gui/*)

add_library(csp-visual-query SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${NODE_FILES}
  ${RESOURCE_FILES}

)

target_link_libraries(csp-visual-query
  PUBLIC
    cs-core
    csl-node-editor
    csl-ogc
)

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-visual-query PROPERTY FOLDER "plugins")

# We mark all resource files as "header" in order to make sure that no one tries to compile them.
set_source_files_properties(${RESOURCE_FILES} PROPERTIES HEADER_FILE_ONLY true)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES} ${RESOURCE_FILES}
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS csp-visual-query DESTINATION "share/plugins")
install(DIRECTORY "gui" DESTINATION "share/resources")
install(FILES   ${NODE_FILES}         DESTINATION "share/resources/nodes/csp-visual-query")

