# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

if (UNIX)
    # Locate header.
    find_path(GDAL_INCLUDE_DIR gdal.h
        HINTS ${GDAL_ROOT_DIR}/include/gdal)

    # Locate libraries.
    find_library(GDAL_LIBRARY NAMES gdal
        HINTS ${GDAL_ROOT_DIR}/lib)
endif()

if (MSVC)
    # Locate header.
    find_path(GDAL_INCLUDE_DIR gdal.h
        HINTS ${GDAL_ROOT_DIR}/include)

    # Locate libraries.
    find_library(GDAL_LIBRARY NAMES gdal_i
        HINTS ${GDAL_ROOT_DIR}/lib)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GDAL DEFAULT_MSG GDAL_INCLUDE_DIR GDAL_LIBRARY)

# Add imported target.
if(GDAL_FOUND)
    set(GDAL_INCLUDE_DIRS "${GDAL_INCLUDE_DIR}")
    set(GDAL_LIBRARIES "${GDAL_LIBRARY}")

    if(NOT GDAL_FIND_QUIETLY)
        message(STATUS "GDAL_INCLUDE_DIRS ........... ${GDAL_INCLUDE_DIR}")
        message(STATUS "GDAL_LIBRARY ................ ${GDAL_LIBRARY}")
    endif(NOT GDAL_FIND_QUIETLY)

    if(NOT TARGET GDAL::GDAL)
        add_library(GDAL::GDAL UNKNOWN IMPORTED)
        set_target_properties(GDAL::GDAL PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${GDAL_INCLUDE_DIRS}")

        set_property(TARGET GDAL::GDAL APPEND PROPERTY
                IMPORTED_LOCATION "${GDAL_LIBRARY}")
    endif()
endif()