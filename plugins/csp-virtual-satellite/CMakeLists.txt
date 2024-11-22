# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSP_VIRTUAL_SATELLITE "Enable compilation of this plugin" ON)

if (NOT CSP_VIRTUAL_SATELLITE)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)
file(GLOB_RECURSE RESOURCE_FILES gui/*)

add_library(csp-virtual-satellite SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOURCE_FILES}
)

target_link_libraries(csp-virtual-satellite
  PUBLIC
    cs-core
)

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-virtual-satellite PROPERTY FOLDER "plugins")

# We mark all resource files as "header" in order to make sure that no one tries to compile them.
set_source_files_properties(${RESOURCE_FILES} PROPERTIES HEADER_FILE_ONLY true)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES} ${RESOURCE_FILES}
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS csp-virtual-satellite DESTINATION "share/plugins")
install(DIRECTORY "gui" DESTINATION "share/resources")
