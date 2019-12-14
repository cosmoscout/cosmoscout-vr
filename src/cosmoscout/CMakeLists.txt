# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# build executable ---------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp */*.cpp)

# Resoucre files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES *.hpp */*.hpp)
file(GLOB_RECURSE RESOUCRE_FILES ../../resources/*)
file(GLOB_RECURSE CONFIG_FILES ../../config/*)

add_executable(cosmoscout
  ${SOURCE_FILES}
  ${HEADER_FILES}
  ${RESOUCRE_FILES}
  ${CONFIG_FILES}
)

# cef needs to be linked first due to https://magpcss.org/ceforum/viewtopic.php?f=6&t=15118&start=10
# to make sure the link order is kept, we need --no-as-needed
set_target_properties(cosmoscout PROPERTIES LINK_WHAT_YOU_USE on)

target_link_libraries(cosmoscout
  cef::cef
  cs-core
)

# We mark all resource and config files as "header"
# in order to make sure that no one tries to compile them.
set_source_files_properties(${RESOUCRE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(${CONFIG_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" PREFIX "src"
  FILES ${SOURCE_FILES} ${HEADER_FILES}
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/../../resources" PREFIX "resources"
  FILES ${RESOUCRE_FILES}
)

source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}/../../config" PREFIX "config"
  FILES ${CONFIG_FILES}
)

# Make sure that CosmoScout VR can be directly started from within Visual Studio.
set_target_properties(cosmoscout PROPERTIES 
  VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_INSTALL_PREFIX}\\bin"
  VS_DEBUGGER_ENVIRONMENT "PATH=..\\lib;%PATH% VISTACORELIBS_DRIVER_PLUGIN_DIRS=..\\lib\\DriverPlugins"
)

# install executable ---------------------------------------------------------------------------------

install(
  TARGETS cosmoscout
  RUNTIME DESTINATION "bin"
)
