# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(GLM_INCLUDE_DIR glm/glm.hpp
    HINTS ${GLM_ROOT_DIR}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLM DEFAULT_MSG GLM_INCLUDE_DIR)

# Add imported target.
if(GLM_FOUND)
    set(GLM_INCLUDE_DIRS "${GLM_INCLUDE_DIR}")

    if(NOT GLM_FIND_QUIETLY)
        message(STATUS "GLM_INCLUDE_DIRS .............. ${GLM_INCLUDE_DIR}")
    endif()

    if(NOT TARGET glm::glm)
        add_library(glm::glm INTERFACE IMPORTED)
        set_target_properties(glm::glm PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${GLM_INCLUDE_DIRS}")
    endif()
endif()
