# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

option(CSP_TRAJECTORIES "Enable compilation of this plugin" ON)

if (NOT CSP_TRAJECTORIES)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp)

# Resoucre files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)
file(GLOB_RECURSE RESOUCRE_FILES gui/*)

add_library(csp-trajectories SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOUCRE_FILES}
)

target_link_libraries(csp-trajectories
  PUBLIC
    cs-core
)

if(COSMOSCOUT_USE_PRECOMPILED_HEADERS)
  target_precompile_headers(csp-trajectories PRIVATE src/precompiled.pch)
endif()

# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-trajectories PROPERTY FOLDER "plugins")

# We mark all resource files as "header" in order to make sure that no one tries to compile them.
set_source_files_properties(${RESOUCRE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES} ${RESOUCRE_FILES}
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS   csp-trajectories   DESTINATION "share/plugins")
install(DIRECTORY "gui"              DESTINATION "share/resources")