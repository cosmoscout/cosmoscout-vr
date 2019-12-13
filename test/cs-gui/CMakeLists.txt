# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# build library ------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp */*.cpp)

add_library(cs-gui-test SHARED
  ${SOURCE_FILES}
)

target_link_libraries(cs-gui-test
  PUBLIC
    cef::cef
    cef::wrapper
    cs-utils
)

install(TARGETS cs-gui-test DESTINATION lib)