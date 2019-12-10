# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# build executable ---------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp */*.cpp)

add_executable(cosmoscout ${SOURCE_FILES})

# cef needs to be linked first due to https://magpcss.org/ceforum/viewtopic.php?f=6&t=15118&start=10
# to make sure the link order is kept, we need --no-as-needed
set_target_properties(cosmoscout PROPERTIES LINK_WHAT_YOU_USE on)

target_link_libraries(cosmoscout
  cef::cef
  cs-core
)

# install executable ---------------------------------------------------------------------------------

install(
  TARGETS cosmoscout
  RUNTIME DESTINATION "bin"
)
