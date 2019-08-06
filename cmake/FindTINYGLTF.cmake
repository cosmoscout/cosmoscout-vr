# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(TINYGLTF_INCLUDE_DIR tiny_gltf.h
    HINTS ${TINYGLTF_ROOT_DIR}/include)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TINYGLTF DEFAULT_MSG TINYGLTF_INCLUDE_DIR)

# Add imported target.
if(TINYGLTF_FOUND)
    set(TINYGLTF_INCLUDE_DIRS "${TINYGLTF_INCLUDE_DIR}")

    if(NOT TINYGLTF_FIND_QUIETLY)
        message(STATUS "TINYGLTF_INCLUDE_DIRS ......... ${TINYGLTF_INCLUDE_DIR}")
    endif()

    if(NOT TARGET tinygltf::tinygltf)
        add_library(tinygltf::tinygltf INTERFACE IMPORTED)
        set_target_properties(tinygltf::tinygltf PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${TINYGLTF_INCLUDE_DIRS}")
    endif()
endif()
