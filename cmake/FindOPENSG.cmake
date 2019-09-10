# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(OPENSG_INCLUDE_DIR OSGAction.h
    HINTS ${OPENSG_ROOT_DIR}/include/OpenSG)

# Locate libraries.
find_library(OPENSG_BASE_LIBRARY NAMES OSGBase OSGBaseD
    HINTS ${OPENSG_ROOT_DIR}/lib ${OPENSG_ROOT_DIR}/lib64)

find_library(OPENSG_SYSTEM_LIBRARY NAMES OSGSystem OSGSystemD
    HINTS ${OPENSG_ROOT_DIR}/lib ${OPENSG_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPENSG DEFAULT_MSG OPENSG_INCLUDE_DIR OPENSG_BASE_LIBRARY OPENSG_SYSTEM_LIBRARY)

# Add imported target.
if(OPENSG_FOUND)
    set(OPENSG_INCLUDE_DIRS "${OPENSG_INCLUDE_DIR}")
    set(OPENSG_LIBRARIES "${OPENSG_BASE_LIBRARY}" "${OPENSG_SYSTEM_LIBRARY}")

    if(NOT OPENSG_FIND_QUIETLY)
        message(STATUS "OPENSG_INCLUDE_DIRS ........... ${OPENSG_INCLUDE_DIR}")
        message(STATUS "OPENSG_BASE_LIBRARY ........... ${OPENSG_BASE_LIBRARY}")
        message(STATUS "OPENSG_SYSTEM_LIBRARY ......... ${OPENSG_SYSTEM_LIBRARY}")
    endif(NOT OPENSG_FIND_QUIETLY)

    if(NOT TARGET OpenSG::base)
        add_library(OpenSG::base UNKNOWN IMPORTED)
        set_target_properties(OpenSG::base PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OPENSG_INCLUDE_DIRS}")

        set_property(TARGET OpenSG::base APPEND PROPERTY
            IMPORTED_LOCATION "${OPENSG_BASE_LIBRARY}")
    endif()

    if(NOT TARGET OpenSG::system)
        add_library(OpenSG::system UNKNOWN IMPORTED)
        set_target_properties(OpenSG::system PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OPENSG_INCLUDE_DIRS}")

        set_property(TARGET OpenSG::system APPEND PROPERTY
            IMPORTED_LOCATION "${OPENSG_SYSTEM_LIBRARY}")
    endif()
endif()
