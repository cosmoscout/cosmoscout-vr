# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# Locate header.
find_path(SNDFILE_INCLUDE_DIR sndfile.h
    HINTS $ENV{LIBSNDFILE_DIR}/include)


find_library(SNDFILE_LIBRARY
    NAMES sndfile   
    HINTS $ENV{LIBSNDFILE_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SndFile DEFAULT_MSG SNDFILE_INCLUDE_DIR SNDFILE_LIBRARY)

# Add imported target.
if(SndFile_FOUND)
    set(SNDFILE_INCLUDE_DIRS "${SNDFILE_INCLUDE_DIR}")

    if(NOT SndFile_FIND_QUIETLY)
        message(STATUS "SNDFILE_INCLUDE_DIRS .............. ${SNDFILE_INCLUDE_DIR}")
    endif()

    if(NOT TARGET sndfile::sndfile)
        add_library(sndfile::sndfile INTERFACE IMPORTED)
        set_target_properties(sndfile::sndfile PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${SNDFILE_INCLUDE_DIRS}")
    endif()
endif()
