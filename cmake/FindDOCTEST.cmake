# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(DOCTEST_INCLUDE_DIR doctest/doctest.h
    HINTS ${DOCTEST_ROOT_DIR}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DOCTEST DEFAULT_MSG DOCTEST_INCLUDE_DIR)

# Add imported target.
if(DOCTEST_FOUND)
    set(DOCTEST_INCLUDE_DIRS "${DOCTEST_INCLUDE_DIR}")

    if(NOT DOCTEST_FIND_QUIETLY)
        message(STATUS "DOCTEST_INCLUDE_DIRS .......... ${DOCTEST_INCLUDE_DIR}")
    endif()

    if(NOT TARGET doctest::doctest)
        add_library(doctest::doctest INTERFACE IMPORTED)
        set_target_properties(doctest::doctest PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${DOCTEST_INCLUDE_DIRS}")
    endif()
endif()
