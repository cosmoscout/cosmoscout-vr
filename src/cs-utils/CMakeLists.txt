# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# build library ------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES *.cpp */*.cpp)

add_library(cs-utils SHARED
  ${SOURCE_FILES}
)

target_link_libraries(cs-utils
  PUBLIC
    glm::glm
    tinygltf::tinygltf
    Boost::filesystem
    Boost::chrono
    Boost::date_time
    GLEW::GLEW
    GLUT::GLUT
    OpenGL::GL
    OpenSG::base
    OpenSG::system
    Tiff::Tiff
    ${VISTACORELIBS_LIBRARIES}
    ${CMAKE_DL_LIBS}
    cspice::cspice
    curl::curl
    curlpp::curlpp
    c-ares::c-ares
)

target_include_directories(cs-utils
  PUBLIC
    ${VISTACORELIBS_INCLUDE_DIRS}
)

install(TARGETS cs-utils DESTINATION lib)

# export header ------------------------------------------------------------------------------------

generate_export_header(cs-utils
  EXPORT_FILE_NAME cs_utils_export.hpp
)
