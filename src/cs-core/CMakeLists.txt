# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# build library ------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp */*.cpp)

add_library(cs-core SHARED
  ${SOURCE_FILES}
)

target_link_libraries(cs-core
  PUBLIC
    cs-scene
    cs-graphics
    cs-gui
    cs-utils
)

target_include_directories(cs-core
  PUBLIC
    ${VISTACORELIBS_INCLUDE_DIRS}
)

install(TARGETS cs-core DESTINATION lib)

# export header ------------------------------------------------------------------------------------

generate_export_header(cs-core
  EXPORT_FILE_NAME cs_core_export.hpp
)
