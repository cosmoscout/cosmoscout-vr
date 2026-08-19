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

find_package(CURL CONFIG REQUIRED)
find_package(cesium-native CONFIG REQUIRED)

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
