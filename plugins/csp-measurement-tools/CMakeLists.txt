# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSP_MEASUREMENT_TOOLS "Enable compilation of this plugin" ON)

if (NOT CSP_MEASUREMENT_TOOLS)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp src/voronoi/*.cpp)

# Resoucre files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp src/voronoi/*.hpp)
file(GLOB_RECURSE RESOUCRE_FILES gui/*)

add_library(csp-measurement-tools SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOUCRE_FILES}
)

target_link_libraries(csp-measurement-tools
  PUBLIC
    cs-core
    csl-tools
)

if (COSMOSCOUT_USE_PRECOMPILED_HEADERS)
  target_precompile_headers(csp-measurement-tools PRIVATE src/precompiled.pch)
endif()

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-measurement-tools PROPERTY FOLDER "plugins")

# We mark all resource files as "header" in order to make sure that no one tries to compile them.
set_source_files_properties(${RESOUCRE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES 
  ${SOURCE_FILES} ${HEADER_FILES} ${RESOUCRE_FILES}
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS    csp-measurement-tools DESTINATION "share/plugins")
install(DIRECTORY "gui"                  DESTINATION "share/resources")
