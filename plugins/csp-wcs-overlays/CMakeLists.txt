# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

option(CSP_WCS_OVERLAYS "Enable compilation of this plugin" ON)

if (NOT CSP_WCS_OVERLAYS)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

if (DEFINED ENV{GDAL_DIR})
  SET(GDAL_DIR "$ENV{GDAL_DIR}")
else()
  if (MSVC)
    SET(GDAL_DIR ${COSMOSCOUT_EXTERNALS_DIR}/gdal)
  endif()

  if(UNIX)
    SET(GDAL_DIR ${COSMOSCOUT_EXTERNALS_DIR})
    SET(GDAL_INCLUDE_DIR "${COSMOSCOUT_EXTERNALS_DIR}/include")
    SET(GDAL_LIBRARY "${COSMOSCOUT_EXTERNALS_DIR}/lib/libgdal.so")
  endif()
endif()

find_package(GDAL REQUIRED)

file(GLOB SOURCE_FILES src/*.cpp)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)
file(GLOB_RECURSE RESOURCE_FILES gui/*)


# GDAL Rendering
file(GLOB CSP_WCS_COMMON_SOURCE_FILES src/common/*.cpp)
file(GLOB CSP_WCS_COMMON_HEADER_FILES src/common/*.hpp)
source_group(rendering FILES ${CSP_WCS_COMMON_SOURCE_FILES} ${CSP_WCS_COMMON_HEADER_FILES})

add_library(csp-wcs-overlays SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOURCE_FILES}
  ${CSP_WCS_COMMON_SOURCE_FILES}
  ${CSP_WCS_COMMON_HEADER_FILES}
)

target_link_libraries(csp-wcs-overlays
  PUBLIC
    cs-core
    GDAL::GDAL
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
