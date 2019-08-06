# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(GLUT_INCLUDE_DIR GL/freeglut.h
    HINTS ${GLUT_ROOT_DIR}/include)

# Locate library.
find_library(GLUT_LIBRARY NAMES glut freeglut
    HINTS ${GLUT_ROOT_DIR}/lib ${GLUT_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLUT DEFAULT_MSG GLUT_INCLUDE_DIR GLUT_LIBRARY)

# Add imported target.
if(GLUT_FOUND)
    set(GLUT_INCLUDE_DIRS "${GLUT_INCLUDE_DIR}")

    if(NOT GLUT_FIND_QUIETLY)
        message(STATUS "GLUT_INCLUDE_DIRS ............. ${GLUT_INCLUDE_DIR}")
        message(STATUS "GLUT_LIBRARY .................. ${GLUT_LIBRARY}")
    endif()

    if(NOT TARGET GLUT::GLUT)
        add_library(GLUT::GLUT UNKNOWN IMPORTED)
        set_target_properties(GLUT::GLUT PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GLUT_INCLUDE_DIRS}")

        set_property(TARGET GLUT::GLUT APPEND PROPERTY
            IMPORTED_LOCATION "${GLUT_LIBRARY}")
    endif()
endif()
