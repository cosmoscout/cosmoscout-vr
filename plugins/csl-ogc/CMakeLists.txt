# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSL_OGC "Enable compilation of this plugin library" ON)

if (NOT CSL_OGC)
  return()
endif()

if (DEFINED ENV{GDAL_ROOT_DIR})
  SET(GDAL_ROOT_DIR "$ENV{GDAL_ROOT_DIR}")
endif()

find_package(GDAL)

if (NOT GDAL_FOUND)
  message(WARNING
          "GDAL was not found; skipping the OGC plugin library!\n"
          "If you wish to use this plugin library please download the GDAL development kit from here: "
          "https://download.gisinternals.com/sdk.php, and set GDAL_ROOT_DIR to the root of the downloaded directory.")
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp src/*/*.cpp)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp src/*/*.hpp)

add_library(csl-ogc SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

target_link_libraries(csl-ogc
  PUBLIC
    cs-core
    GDAL::GDAL
)

# Add this plugin library to a "plugins" folder in your IDE.
set_property(TARGET csl-ogc PROPERTY FOLDER "plugins")

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES}
)

target_include_directories(csl-ogc PUBLIC
  ${CMAKE_BINARY_DIR}/plugins/csl-ogc
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS csl-ogc DESTINATION "lib")
install(DIRECTORY "${GDAL_ROOT_DIR}/bin" DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(DIRECTORY "${GDAL_ROOT_DIR}/lib" DESTINATION "${CMAKE_INSTALL_PREFIX}")

# export header ------------------------------------------------------------------------------------

generate_export_header(csl-ogc
    EXPORT_FILE_NAME csl_ogc_export.hpp
)
