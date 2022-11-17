# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSP_WCS_OVERLAYS "Enable compilation of this plugin" ON)

if (NOT CSP_WCS_OVERLAYS)
  return()
endif()

if (NOT CSL_OGC)
  message(WARNING "The OGC plugin library wasn't found! Skipping csp-wcs-overlays.")
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)
file(GLOB_RECURSE RESOURCE_FILES gui/*)

add_library(csp-wcs-overlays SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOURCE_FILES}
)

target_link_libraries(csp-wcs-overlays
  PUBLIC
    cs-core
    csl-ogc
)

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-wcs-overlays PROPERTY FOLDER "plugins")

# We mark all resource files as "header" in order to make sure that no one tries to compile them.
set_source_files_properties(${RESOUCRE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES} ${RESOUCRE_FILES}
)

# Define include directories
target_include_directories(csp-wcs-overlays
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/..
)



# install plugin -----------------------------------------------------------------------------------

install(TARGETS   csp-wcs-overlays DESTINATION "share/plugins")
install(DIRECTORY "gui"            DESTINATION "share/resources")
