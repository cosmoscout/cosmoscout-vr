# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(CEF_INCLUDE_DIR include/cef_version.h
    HINTS ${CEF_ROOT_DIR}/include/cef)

# Locate library.
find_path(CEF_RESOURCE_DIR cef.pak
    HINTS ${CEF_ROOT_DIR}/share/cef)

find_path(CEF_LIBRARY_DIR snapshot_blob.bin
    HINTS ${CEF_ROOT_DIR}/lib)

find_library(CEF_LIBRARY NAMES cef libcef
    HINTS ${CEF_ROOT_DIR}/lib)

find_library(CEF_WRAPPER_LIBRARY NAMES cef_dll_wrapper.a libcef_dll_wrapper.a libcef_dll_wrapper.lib
    HINTS ${CEF_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CEF DEFAULT_MSG CEF_LIBRARY CEF_WRAPPER_LIBRARY CEF_INCLUDE_DIR CEF_LIBRARY_DIR)

# Add imported target.
if(CEF_FOUND)
    set(CEF_INCLUDE_DIRS "${CEF_INCLUDE_DIR}")

    if(NOT CEF_FIND_QUIETLY)
        message(STATUS "CEF_RESOURCE_DIR .............. ${CEF_RESOURCE_DIR}")
        message(STATUS "CEF_LIBRARY_DIR ............... ${CEF_LIBRARY_DIR}")
        message(STATUS "CEF_INCLUDE_DIRS .............. ${CEF_INCLUDE_DIR}")
        message(STATUS "CEF_LIBRARY ................... ${CEF_LIBRARY}")
        message(STATUS "CEF_WRAPPER_LIBRARY ........... ${CEF_WRAPPER_LIBRARY}")
    endif()

    if(NOT TARGET cef::cef)
        add_library(cef::cef UNKNOWN IMPORTED)
        set_target_properties(cef::cef PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CEF_INCLUDE_DIRS}")

        set_property(TARGET cef::cef APPEND PROPERTY
            IMPORTED_LOCATION "${CEF_LIBRARY}")
    endif()

    if(NOT TARGET cef::wrapper)
        add_library(cef::wrapper UNKNOWN IMPORTED)
        set_target_properties(cef::wrapper PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${CEF_INCLUDE_DIRS}")

        set_property(TARGET cef::wrapper APPEND PROPERTY
            IMPORTED_LOCATION "${CEF_WRAPPER_LIBRARY}")
    endif()
endif()
