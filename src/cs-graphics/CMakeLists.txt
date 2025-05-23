# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# build library ------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp */*.cpp)

# Header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES *.hpp */*.hpp)

# Shader files are also only added in order to make them available in your IDE.
set(RESOURCE_SHADER_DIR ${CMAKE_SOURCE_DIR}/resources/shaders/)
file(GLOB SHADER_FILES ${RESOURCE_SHADER_DIR}* ${RESOURCE_SHADER_DIR}*/*)

source_group("src/shaders" FILES ${SHADER_FILES})

add_library(cs-graphics SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${SHADER_FILES}
)

target_link_libraries(cs-graphics
  PUBLIC
    gli::gli
    cs-utils
)

if(COSMOSCOUT_USE_PRECOMPILED_HEADERS)
  target_precompile_headers(cs-graphics PRIVATE precompiled.pch)
endif()

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "src" FILES
  ${SOURCE_FILES} ${HEADER_FILES}
)

# install the library ------------------------------------------------------------------------------

install(TARGETS cs-graphics DESTINATION lib)

# export header ------------------------------------------------------------------------------------

generate_export_header(cs-graphics
  EXPORT_FILE_NAME cs_graphics_export.hpp
)
