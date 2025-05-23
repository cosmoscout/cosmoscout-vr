# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSP_STARS "Enable compilation of this plugin" ON)

if (NOT CSP_STARS)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp)

# Resoucre files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)
file(GLOB_RECURSE RESOUCRE_FILES gui/* textures/*)

# Shader files are also only added in order to make them available in your IDE.
set(CSP_STARS_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders/)
file(GLOB SHADER_FILES ${CSP_STARS_SHADER_DIR}* ${CSP_STARS_SHADER_DIR}*/*)

source_group("src/shaders" FILES ${SHADER_FILES})

add_library(csp-stars SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOUCRE_FILES}
  ${SHADER_FILES}
)

target_link_libraries(csp-stars
  PUBLIC
    cs-core
)

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-stars PROPERTY FOLDER "plugins")

# We mark all resource files as "header" in order to make sure that no one tries to compile them.
set_source_files_properties(${RESOUCRE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES 
  ${SOURCE_FILES} ${HEADER_FILES} ${RESOUCRE_FILES}
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS csp-stars    DESTINATION "share/plugins")
install(DIRECTORY "gui"      DESTINATION "share/resources")
install(DIRECTORY "shaders"  DESTINATION "share/resources")
install(DIRECTORY "textures" DESTINATION "share/resources")