# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(TIFF_INCLUDE_DIR tiff.h
    HINTS ${TIFF_ROOT_DIR}/include)

# Locate library.
find_library(TIFF_LIBRARY NAMES tiff tiffd
    HINTS ${TIFF_ROOT_DIR}/lib ${TIFF_ROOT_DIR}/lib64)
find_library(TIFFXX_LIBRARY NAMES tiffxx tiffxxd
    HINTS ${TIFF_ROOT_DIR}/lib ${TIFF_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TIFF DEFAULT_MSG TIFF_INCLUDE_DIR TIFF_LIBRARY TIFFXX_LIBRARY)

# Add imported target.
if(TIFF_FOUND)
    set(TIFF_INCLUDE_DIRS "${TIFF_INCLUDE_DIR}")

    if(NOT TIFF_FIND_QUIETLY)
        message(STATUS "TIFF_INCLUDE_DIRS ............. ${TIFF_INCLUDE_DIR}")
        message(STATUS "TIFF_LIBRARY .................. ${TIFF_LIBRARY}")
        message(STATUS "TIFFXX_LIBRARY ................ ${TIFFXX_LIBRARY}")
    endif()

    if(NOT TARGET Tiff::Tiff)
        add_library(Tiff::Tiff UNKNOWN IMPORTED)
        set_target_properties(Tiff::Tiff PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TIFF_INCLUDE_DIRS}")

        set_property(TARGET Tiff::Tiff APPEND PROPERTY
            IMPORTED_LOCATION "${TIFF_LIBRARY}")
    endif()

    if(NOT TARGET Tiff::Tiffxx)
        add_library(Tiff::Tiffxx UNKNOWN IMPORTED)
        set_target_properties(Tiff::Tiffxx PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TIFF_INCLUDE_DIRS}")

        set_property(TARGET Tiff::Tiffxx APPEND PROPERTY
            IMPORTED_LOCATION "${TIFFXX_LIBRARY}")
    endif()
endif()
