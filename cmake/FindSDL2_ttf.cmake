# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# Locate header.
find_path(SDL2_TTF_INCLUDE_DIR SDL2/SDL_ttf.h
    HINTS ${SDL2_TTF_ROOT_DIR}/include)

# Locate library.
find_library(SDL2_TTF_LIBRARY NAMES SDL2_ttf SDL2_ttfd
    HINTS ${SDL2_TTF_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SDL2_ttf DEFAULT_MSG SDL2_TTF_INCLUDE_DIR SDL2_TTF_LIBRARY)

# Add imported target.
if(SDL2_TTF_FOUND)
    set(SDL2_TTF_INCLUDE_DIRS "${SDL2_TTF_INCLUDE_DIR}")

    if(NOT SDL2_TTF_FIND_QUIETLY)
        message(STATUS "SDL2_TTF_INCLUDE_DIRS ......... ${SDL2_TTF_INCLUDE_DIR}")
        message(STATUS "SDL2_TTF_LIBRARY .............. ${SDL2_TTF_LIBRARY}")
    endif()

    if(NOT TARGET SDL2::ttf)
        add_library(SDL2::ttf UNKNOWN IMPORTED)
        set_target_properties(SDL2::ttf PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${SDL2_TTF_INCLUDE_DIRS}")

        set_property(TARGET SDL2::ttf APPEND PROPERTY
            IMPORTED_LOCATION "${SDL2_TTF_LIBRARY}")
    endif()
endif()
