# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# cef-config.cmake
#
# Provides:
#   cef::libcef              - IMPORTED SHARED, the prebuilt Chromium runtime
#   cef::libcef_dll_wrapper  - IMPORTED STATIC, the wrapper built from source
#                              (link this one; it pulls in cef::libcef)
#   CEF_RESOURCE_DIR         - path to Resources/ (*.pak files, locales/)
#   CEF_BINARY_DIR           - path to bin/ (libcef and friends)
#   cef_copy_runtime_files(<target>) - post-build step that copies the DLLs
#                              and resources next to <target>'s output binary

get_filename_component(_cef_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

if(NOT TARGET cef::libcef)
    find_file(CEF_LIBCEF_DLL NAMES libcef.dll libcef.so PATHS "${_cef_root}/bin" NO_DEFAULT_PATH)
    find_library(CEF_LIBCEF_LIB NAMES libcef PATHS "${_cef_root}/lib" NO_DEFAULT_PATH)

    if(NOT CEF_LIBCEF_DLL OR ((NOT CEF_LIBCEF_LIB) AND WIN32))
        message(FATAL_ERROR "cef-config.cmake: could not locate libcef libraries under ${_cef_root}")
    endif()

    add_library(cef::libcef SHARED IMPORTED)
    set_target_properties(cef::libcef PROPERTIES
        IMPORTED_LOCATION "${CEF_LIBCEF_DLL}"
        IMPORTED_IMPLIB "${CEF_LIBCEF_LIB}"
        INTERFACE_INCLUDE_DIRECTORIES "${_cef_root}/include/cef"
    )
endif()

if(NOT TARGET cef::libcef_dll_wrapper)
    find_library(CEF_WRAPPER_LIB_RELEASE
        NAMES
        libcef_dll_wrapper
        libcef_dll_wrapper.a
        libcef_dll_wrapper.lib
        PATHS "${_cef_root}/lib" NO_DEFAULT_PATH)

    if(NOT CEF_WRAPPER_LIB_RELEASE)
        message(FATAL_ERROR "cef-config.cmake: could not locate libcef_dll_wrapper library under ${_cef_root}/lib")
    endif()

    find_library(CEF_WRAPPER_LIB_DEBUG
        NAMES
        libcef_dll_wrapper
        libcef_dll_wrapper.a
        libcef_dll_wrapper.lib
        PATHS "${_cef_root}/debug/lib" NO_DEFAULT_PATH)

    add_library(cef::libcef_dll_wrapper STATIC IMPORTED)
    set_target_properties(cef::libcef_dll_wrapper PROPERTIES
        IMPORTED_LOCATION "${CEF_WRAPPER_LIB_RELEASE}"
        IMPORTED_LOCATION_RELEASE "${CEF_WRAPPER_LIB_RELEASE}"
        IMPORTED_LOCATION_RELWITHDEBINFO "${CEF_WRAPPER_LIB_RELEASE}"
        IMPORTED_LOCATION_MINSIZEREL "${CEF_WRAPPER_LIB_RELEASE}"
        INTERFACE_LINK_LIBRARIES "cef::libcef"
    )

    if(CEF_WRAPPER_LIB_DEBUG)
        set_property(TARGET cef::libcef_dll_wrapper PROPERTY
            IMPORTED_LOCATION_DEBUG "${CEF_WRAPPER_LIB_DEBUG}")
    endif()
endif()

set(CEF_RESOURCE_DIR "${_cef_root}/share/cef/Resources")
set(CEF_BINARY_DIR "${_cef_root}/bin")

# Copies libcef + the rest of the Chromium runtime, plus Resources/
# (*.pak files and locales/), next to <target>'s built binary as a post-build
# step. Call this once per executable/plugin target that links cef::libcef.
function(cef_copy_runtime_files target)
    if(NOT TARGET ${target})
        message(FATAL_ERROR "cef_copy_runtime_files: '${target}' is not a CMake target")
    endif()

    file(GLOB _cef_runtime_files
        "${CEF_BINARY_DIR}/*.dll"
        "${CEF_BINARY_DIR}/*.so"
        "${CEF_BINARY_DIR}/*.so.1"
        "${CEF_BINARY_DIR}/*.dat"
        "${CEF_BINARY_DIR}/*.bin"
        "${CEF_BINARY_DIR}/*.json"
    )

    foreach(_f IN LISTS _cef_runtime_files)
        add_custom_command(TARGET ${target} POST_BUILD
            COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${_f}" "$<TARGET_FILE_DIR:${target}>"
        )
    endforeach()

    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CEF_RESOURCE_DIR}" "$<TARGET_FILE_DIR:${target}>"
    )
endfunction()
