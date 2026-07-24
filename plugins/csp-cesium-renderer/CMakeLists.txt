# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# 1. The Toggle: Allows developers to turn off compilation of this specific plugin
option(CSP_CESIUM_RENDERER "Enable compilation of this plugin" ON) 

if(NOT CSP_CESIUM_RENDERER)
    return()
endif()

if (DEFINED ENV{cesium-native_DIR})
    SET(cesium-native_DIR "$ENV{cesium-native_DIR}")
else()
    SET(cesium-native_DIR ${COSMOSCOUT_EXTERNALS_DIR}/share/cesium-native/cmake)
endif()

if (DEFINED ENV{ZLIB_ROOT})
    SET(ZLIB_ROOT "$ENV{ZLIB_ROOT}")
else()
    SET(ZLIB_ROOT ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{ZLIB_DIR})
    SET(ZLIB_DIR "$ENV{ZLIB_DIR}")
else()
    SET(ZLIB_DIR ${COSMOSCOUT_EXTERNALS_DIR}/share/zlib)
endif()

# spz uses CMake's FindZLIB module and expects its ZLIB::ZLIB target. The static Windows
# externals use the vcpkg names zs.lib and zsd.lib, which FindZLIB does not search for.
if(WIN32)
    set(ZLIB_LIBRARY_RELEASE "${ZLIB_ROOT}/lib/zs.lib")
    set(ZLIB_LIBRARY_DEBUG "${ZLIB_ROOT}/debug/lib/zsd.lib")
endif()

# Load the externals package before cesium-native's module-mode dependency lookup. This supplies
# CURL::libcurl_static, which is part of CesiumCurl's exported link interface.
set(CURL_DIR "${COSMOSCOUT_EXTERNALS_DIR}/share/curl")
find_package(CURL CONFIG REQUIRED)
find_package(cesium-native REQUIRED)

# build plugin -------------------------------------------------------------------------------------

# 2. Define Files used in this Plugin Here (Did not use GLOB becasue it created ghost files which compiler missed)
set(SOURCE_FILES
  src/Plugin.cpp
  src/logger.cpp
  src/CesiumUtils.cpp
  src/CesiumTilesetRenderer.cpp
)

set(HEADER_FILES
  src/Plugin.hpp
  src/logger.hpp
  src/CesiumUtils.hpp
  src/CesiumTilesetRenderer.hpp
)

# 3. Create Target
add_library(csp-cesium-renderer SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
)


# 4. Link CosmoScout Core Libraries
target_link_libraries(csp-cesium-renderer
  PUBLIC
    cs-core
  PRIVATE
    Cesium3DTilesSelection
    Cesium3DTilesContent
    CesiumAsync
    CesiumCurl
    CesiumGltf
    CesiumGltfContent
    CesiumImage
    CesiumUtility
)

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-cesium-renderer PROPERTY FOLDER "plugins")

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES}
)

# install plugin -----------------------------------------------------------------------------------
install(TARGETS csp-cesium-renderer DESTINATION "share/plugins")
