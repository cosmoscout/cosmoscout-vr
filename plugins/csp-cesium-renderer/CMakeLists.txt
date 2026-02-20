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

# build plugin -------------------------------------------------------------------------------------

# 2. Define Files used in this Plugin Here (Did not use GLOB becasue it created ghost files which compiler missed)
set(SOURCE_FILES
  src/Plugin.cpp
)

set(HEADER_FILES
  src/Plugin.hpp
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
)


# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-cesium-renderer PROPERTY FOLDER "plugins")

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES}
)

# install plugin -----------------------------------------------------------------------------------
install(TARGETS csp-cesium-renderer DESTINATION "share/plugins")


