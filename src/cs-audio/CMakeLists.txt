# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# build library ------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp */*.cpp)

# Header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES *.hpp */*.hpp)

add_library(cs-audio SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

target_link_libraries(cs-audio
  PUBLIC
    openal-soft
    cs-utils
)

if(COSMOSCOUT_USE_PRECOMPILED_HEADERS)
  target_precompile_headers(cs-audio PRIVATE precompiled.pch)
endif()

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "src" FILES
  ${SOURCE_FILES} ${HEADER_FILES}
)

# install the library ------------------------------------------------------------------------------

install(TARGETS cs-audio DESTINATION lib)

# export header ------------------------------------------------------------------------------------

generate_export_header(cs-audio
  EXPORT_FILE_NAME cs_audio_export.hpp
)
