# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# Locate header.
find_path(SDL2_INCLUDE_DIR SDL2/SDL.h
    HINTS ${SDL2_ROOT_DIR}/include)

# Locate library.
find_library(SDL2_LIBRARY NAMES SDL2
    HINTS ${SDL2_ROOT_DIR}/lib)
    
find_library(SDL2_MAIN_LIBRARY NAMES SDL2main
    HINTS ${SDL2_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SDL2 DEFAULT_MSG SDL2_INCLUDE_DIR SDL2_LIBRARY)

# Add imported target.
if(SDL2_FOUND)
    set(SDL2_INCLUDE_DIRS "${SDL2_INCLUDE_DIR}")

    if(NOT SDL2_FIND_QUIETLY)
        message(STATUS "SDL2_INCLUDE_DIRS ............. ${SDL2_INCLUDE_DIR}")
        message(STATUS "SDL2_LIBRARY .................. ${SDL2_LIBRARY}")
        message(STATUS "SDL2_MAIN_LIBRARY ............. ${SDL2_MAIN_LIBRARY}")
    endif()

    if(NOT TARGET SDL2::base)
        add_library(SDL2::base UNKNOWN IMPORTED)
        set_target_properties(SDL2::base PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${SDL2_INCLUDE_DIRS}")

        set_property(TARGET SDL2::base APPEND PROPERTY
            IMPORTED_LOCATION "${SDL2_LIBRARY}")
    endif()
    if(NOT TARGET SDL2::main)
        add_library(SDL2::main UNKNOWN IMPORTED)
        set_target_properties(SDL2::main PROPERTIES
            IMPORTED_LOCATION "${SDL2_MAIN_LIBRARY}")
    endif()
endif()
