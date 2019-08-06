# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(GLI_INCLUDE_DIR gli/gli.hpp
    HINTS ${GLI_ROOT_DIR}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLI DEFAULT_MSG GLI_INCLUDE_DIR)

# Add imported target.
if(GLI_FOUND)
    set(GLI_INCLUDE_DIRS "${GLI_INCLUDE_DIR}")

    if(NOT GLI_FIND_QUIETLY)
        message(STATUS "GLI_INCLUDE_DIRS .............. ${GLI_INCLUDE_DIR}")
    endif()

    if(NOT TARGET gli::gli)
        add_library(gli::gli INTERFACE IMPORTED)
        set_target_properties(gli::gli PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GLI_INCLUDE_DIRS}")
    endif()
endif()
