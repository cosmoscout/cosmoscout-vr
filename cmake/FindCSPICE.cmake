# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(CSPICE_INCLUDE_DIR cspice/SpiceUsr.h
    HINTS ${CSPICE_ROOT_DIR}/include)

# Locate library.
find_library(CSPICE_LIBRARY NAMES cspice.a cspice.lib
    HINTS ${CSPICE_ROOT_DIR}/lib ${CSPICE_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CSPICE DEFAULT_MSG CSPICE_INCLUDE_DIR CSPICE_LIBRARY)

# Add imported target.
if(CSPICE_FOUND)
    set(CSPICE_INCLUDE_DIRS "${CSPICE_INCLUDE_DIR}")

    if(NOT CSPICE_FIND_QUIETLY)
        message(STATUS "CSPICE_INCLUDE_DIRS ........... ${CSPICE_INCLUDE_DIR}")
        message(STATUS "CSPICE_LIBRARY ................ ${CSPICE_LIBRARY}")
    endif()

    if(NOT TARGET cspice::cspice)
        add_library(cspice::cspice UNKNOWN IMPORTED)
        set_target_properties(cspice::cspice PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CSPICE_INCLUDE_DIRS}")

        set_property(TARGET cspice::cspice APPEND PROPERTY
            IMPORTED_LOCATION "${CSPICE_LIBRARY}")
    endif()
endif()
