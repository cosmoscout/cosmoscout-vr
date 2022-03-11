# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(CURLPP_INCLUDE_DIR curlpp/cURLpp.hpp
    HINTS ${CURLPP_ROOT_DIR}/include)

# Locate library.
find_library(CURLPP_LIBRARY NAMES curlpp libcurlpp
    HINTS ${CURLPP_ROOT_DIR}/lib ${CURLPP_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CURLPP DEFAULT_MSG CURLPP_INCLUDE_DIR CURLPP_LIBRARY)

# Add imported target.
if(CURLPP_FOUND)
    set(CURLPP_INCLUDE_DIRS "${CURLPP_INCLUDE_DIR}")

    if(NOT CURLPP_FIND_QUIETLY)
        message(STATUS "CURLPP_INCLUDE_DIRS ........... ${CURLPP_INCLUDE_DIR}")
        message(STATUS "CURLPP_LIBRARY ................ ${CURLPP_LIBRARY}")
    endif()

    if(NOT TARGET curlpp::curlpp)
        add_library(curlpp::curlpp UNKNOWN IMPORTED)
        set_target_properties(curlpp::curlpp PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CURLPP_INCLUDE_DIRS}")

        set_property(TARGET curlpp::curlpp APPEND PROPERTY
            IMPORTED_LOCATION "${CURLPP_LIBRARY}")
    endif()
endif()
