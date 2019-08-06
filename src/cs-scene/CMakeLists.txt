# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# build library ------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp */*.cpp)

add_library(cs-scene SHARED
  ${SOURCE_FILES}
)

target_link_libraries(cs-scene
  PUBLIC
    cs-utils
)

install(TARGETS cs-scene DESTINATION lib)

# export header ------------------------------------------------------------------------------------

generate_export_header(cs-scene
  EXPORT_FILE_NAME cs_scene_export.hpp
)
