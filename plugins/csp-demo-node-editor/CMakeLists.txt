# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSP_DEMO_NODE_EDITOR "Enable compilation of this plugin" ON)

if (NOT CSP_DEMO_NODE_EDITOR)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp src/nodes/*.cpp)

# Resoucre files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)

add_library(csp-demo-node-editor SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

target_link_libraries(csp-demo-node-editor
  PUBLIC
    csl-node-editor
)

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-demo-node-editor PROPERTY FOLDER "plugins")

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES 
  ${SOURCE_FILES} ${HEADER_FILES}
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS   csp-demo-node-editor  DESTINATION "share/plugins")

