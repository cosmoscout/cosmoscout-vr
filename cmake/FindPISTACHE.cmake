# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(PISTACHE_INCLUDE_DIR pistache/endpoint.h
    HINTS ${PISTACHE_ROOT_DIR}/include)

# Locate library.
find_library(PISTACHE_LIBRARY NAMES pistache pistached
    HINTS ${PISTACHE_ROOT_DIR}/lib ${PISTACHE_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PISTACHE DEFAULT_MSG PISTACHE_INCLUDE_DIR PISTACHE_LIBRARY)

# Add imported target.
if(PISTACHE_FOUND)
    set(PISTACHE_INCLUDE_DIRS "${PISTACHE_INCLUDE_DIR}")

    if(NOT PISTACHE_FIND_QUIETLY)
        message(STATUS "PISTACHE_INCLUDE_DIRS ......... ${PISTACHE_INCLUDE_DIR}")
        message(STATUS "PISTACHE_LIBRARY .............. ${PISTACHE_LIBRARY}")
    endif()

    if(NOT TARGET pistache::pistache)
        add_library(pistache::pistache UNKNOWN IMPORTED)
        set_target_properties(pistache::pistache PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${PISTACHE_INCLUDE_DIRS}")

        set_property(TARGET pistache::pistache APPEND PROPERTY
            IMPORTED_LOCATION "${PISTACHE_LIBRARY}")
    endif()
endif()
