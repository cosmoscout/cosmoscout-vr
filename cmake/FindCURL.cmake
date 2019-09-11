# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(CURL_INCLUDE_DIR curl/curl.h
    HINTS ${CURL_ROOT_DIR}/include)

# Locate library.
find_library(CURL_LIBRARY NAMES curl libcurl_imp libcurl-d_imp
    HINTS ${CURL_ROOT_DIR}/lib ${CURL_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CURL DEFAULT_MSG CURL_INCLUDE_DIR CURL_LIBRARY)

# Add imported target.
if(CURL_FOUND)
    set(CURL_INCLUDE_DIRS "${CURL_INCLUDE_DIR}")

    if(NOT CURL_FIND_QUIETLY)
        message(STATUS "CURL_INCLUDE_DIRS ............. ${CURL_INCLUDE_DIR}")
        message(STATUS "CURL_LIBRARY .................. ${CURL_LIBRARY}")
    endif()

    if(NOT TARGET curl::curl)
        add_library(curl::curl UNKNOWN IMPORTED)
        set_target_properties(curl::curl PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CURL_INCLUDE_DIRS}")

        set_property(TARGET curl::curl APPEND PROPERTY
            IMPORTED_LOCATION "${CURL_LIBRARY}")
    endif()
endif()
