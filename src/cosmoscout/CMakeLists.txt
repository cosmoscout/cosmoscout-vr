# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

set(TARGET_NAME cosmoscout)

# build executable ---------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp */*.cpp)

add_executable(${TARGET_NAME} ${SOURCE_FILES})

target_link_libraries(${TARGET_NAME}
  curl::curl
  curlpp::curlpp
  c-ares::c-ares
  cs-core
)

# install executable ---------------------------------------------------------------------------------

install(
  TARGETS ${TARGET_NAME}
  RUNTIME DESTINATION "bin"
)
