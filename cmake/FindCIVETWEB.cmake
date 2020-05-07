# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(CIVETWEB_INCLUDE_DIR CivetServer.h
    HINTS ${CIVETWEB_ROOT_DIR}/include)

# Locate library.
find_library(CIVETWEB_LIBRARY NAMES civetweb
    HINTS ${CIVETWEB_ROOT_DIR}/lib ${CIVETWEB_ROOT_DIR}/lib64)

find_library(CIVETWEBCXX_LIBRARY NAMES civetweb-cpp
    HINTS ${CIVETWEB_ROOT_DIR}/lib ${CIVETWEB_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CIVETWEB DEFAULT_MSG CIVETWEB_INCLUDE_DIR 
    CIVETWEB_LIBRARY CIVETWEBCXX_LIBRARY)

# Add imported target.
if(CIVETWEB_FOUND)
    set(CIVETWEB_INCLUDE_DIRS "${CIVETWEB_INCLUDE_DIR}")

    if(NOT CIVETWEB_FIND_QUIETLY)
        message(STATUS "CIVETWEB_INCLUDE_DIRS ......... ${CIVETWEB_INCLUDE_DIR}")
        message(STATUS "CIVETWEB_LIBRARY .............. ${CIVETWEB_LIBRARY}")
        message(STATUS "CIVETWEBCXX_LIBRARY ........... ${CIVETWEBCXX_LIBRARY}")
    endif()

    if(NOT TARGET civetweb::civetweb)
        add_library(civetweb::civetweb UNKNOWN IMPORTED)
        set_target_properties(civetweb::civetweb PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CIVETWEB_INCLUDE_DIRS}")

        set_property(TARGET civetweb::civetweb APPEND PROPERTY
            IMPORTED_LOCATION "${CIVETWEB_LIBRARY}")
    endif()

    if(NOT TARGET civetweb::civetwebcpp)
        add_library(civetweb::civetwebcpp UNKNOWN IMPORTED)
        set_target_properties(civetweb::civetwebcpp PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CIVETWEB_INCLUDE_DIRS}")

        set_property(TARGET civetweb::civetwebcpp APPEND PROPERTY
            IMPORTED_LOCATION "${CIVETWEBCXX_LIBRARY}")
    endif()
endif()
