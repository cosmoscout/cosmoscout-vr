# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(CARES_INCLUDE_DIR ares.h
    HINTS ${CARES_ROOT_DIR}/include)

# Locate library.
find_library(CARES_LIBRARY NAMES cares
    HINTS ${CARES_ROOT_DIR}/lib ${CARES_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CARES DEFAULT_MSG CARES_INCLUDE_DIR CARES_LIBRARY)

# Add imported target.
if(CARES_FOUND)
    set(CARES_INCLUDE_DIRS "${CARES_INCLUDE_DIR}")

    if(NOT CARES_FIND_QUIETLY)
        message(STATUS "CARES_INCLUDE_DIRS ............ ${CARES_INCLUDE_DIR}")
        message(STATUS "CARES_LIBRARY ................. ${CARES_LIBRARY}")
    endif()

    if(NOT TARGET c-ares::c-ares)
        add_library(c-ares::c-ares UNKNOWN IMPORTED)
        set_target_properties(c-ares::c-ares PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CARES_INCLUDE_DIRS}")

        set_property(TARGET c-ares::c-ares APPEND PROPERTY
            IMPORTED_LOCATION "${CARES_LIBRARY}")
    endif()
endif()
