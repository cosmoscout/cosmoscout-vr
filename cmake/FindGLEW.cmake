# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(GLEW_INCLUDE_DIR GL/glew.h
    HINTS ${GLEW_ROOT_DIR}/include)

# Locate library.
find_library(GLEW_LIBRARY NAMES GLEW glew32
    HINTS ${GLEW_ROOT_DIR}/lib ${GLEW_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLEW DEFAULT_MSG GLEW_INCLUDE_DIR GLEW_LIBRARY)

# Add imported target.
if(GLEW_FOUND)
    set(GLEW_INCLUDE_DIRS "${GLEW_INCLUDE_DIR}")

    if(NOT GLEW_FIND_QUIETLY)
        message(STATUS "GLEW_INCLUDE_DIRS ............. ${GLEW_INCLUDE_DIR}")
        message(STATUS "GLEW_LIBRARY .................. ${GLEW_LIBRARY}")
    endif()

    if(NOT TARGET GLEW::GLEW)
        add_library(GLEW::GLEW UNKNOWN IMPORTED)
        set_target_properties(GLEW::GLEW PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GLEW_INCLUDE_DIRS}")

        set_property(TARGET GLEW::GLEW APPEND PROPERTY
            IMPORTED_LOCATION "${GLEW_LIBRARY}")
    endif()
endif()
