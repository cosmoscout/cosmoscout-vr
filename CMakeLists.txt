# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

cmake_minimum_required(VERSION 3.13)
project(cosmoscout-vr VERSION 1.3.0)

# Use cmake 3.12's <PACKAGE>_ROOT variabled for searching.
cmake_policy(SET CMP0074 NEW)

# Ensure local modules (for dependencies etc.) are found.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# CMAKE_BUILD_TYPE must be set (except for Visual Studio).
if(NOT MSVC)
  if(NOT CMAKE_BUILD_TYPE OR
    (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Release" AND
    NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug"))

    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release or Debug" FORCE)
  endif()
endif()

# Use folders when targeting an IDE
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# find dependencies --------------------------------------------------------------------------------

include(GenerateExportHeader)

# Boost and OpenGL must be present on your system. All other dependencies are included as submodules
# in "externals/". Those must be built beforehand, breferably using the scripts "make_externals.*".
set(Boost_REALPATH ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_LIBS OFF)
find_package(Boost REQUIRED COMPONENTS system chrono date_time filesystem)
find_package(OpenGL REQUIRED)

# You have to provide the directory where the externals got installed to. The scripts make_*.sh and
# make_*.bat set this directory via the command line.
set(COSMOSCOUT_EXTERNALS_DIR COSMOSCOUT_EXTERNALS_DIR-NotFound 
  CACHE STRING "Directory where the externals got installed to.")

if (DEFINED ENV{CARES_ROOT_DIR})
  SET(CARES_ROOT_DIR "$ENV{CARES_ROOT_DIR}")
else()
  SET(CARES_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{CURL_ROOT_DIR})
  SET(CURL_ROOT_DIR "$ENV{CURL_ROOT_DIR}")
else()
  SET(CURL_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{CURLPP_ROOT_DIR})
  SET(CURLPP_ROOT_DIR "$ENV{CURLPP_ROOT_DIR}")
else()
  SET(CURLPP_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{SPDLOG_ROOT_DIR})
  SET(SPDLOG_ROOT_DIR "$ENV{SPDLOG_ROOT_DIR}")
else()
  SET(SPDLOG_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{GLM_ROOT_DIR})
  SET(GLM_ROOT_DIR "$ENV{GLM_ROOT_DIR}")
else()
  SET(GLM_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{GLI_ROOT_DIR})
  SET(GLI_ROOT_DIR "$ENV{GLI_ROOT_DIR}")
else()
  SET(GLI_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{DOCTEST_ROOT_DIR})
  SET(DOCTEST_ROOT_DIR "$ENV{DOCTEST_ROOT_DIR}")
else()
  SET(DOCTEST_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{TINYGLTF_ROOT_DIR})
  SET(TINYGLTF_ROOT_DIR "$ENV{TINYGLTF_ROOT_DIR}")
else()
  SET(TINYGLTF_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{CSPICE_ROOT_DIR})
  SET(CSPICE_ROOT_DIR "$ENV{CSPICE_ROOT_DIR}")
else()
  SET(CSPICE_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{CEF_ROOT_DIR})
  SET(CEF_ROOT_DIR "$ENV{CEF_ROOT_DIR}")
else()
  SET(CEF_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{TIFF_ROOT_DIR})
  SET(TIFF_ROOT_DIR "$ENV{TIFF_ROOT_DIR}")
else()
  SET(TIFF_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{GLEW_ROOT_DIR})
  SET(GLEW_ROOT_DIR "$ENV{GLEW_ROOT_DIR}")
else()
  SET(GLEW_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{GLUT_ROOT_DIR})
  SET(GLUT_ROOT_DIR "$ENV{GLUT_ROOT_DIR}")
else()
  SET(GLUT_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{OPENSG_ROOT_DIR})
  SET(OPENSG_ROOT_DIR "$ENV{OPENSG_ROOT_DIR}")
else()
  SET(OPENSG_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{CIVETWEB_ROOT_DIR})
  SET(CIVETWEB_ROOT_DIR "$ENV{CIVETWEB_ROOT_DIR}")
else()
  SET(CIVETWEB_ROOT_DIR ${COSMOSCOUT_EXTERNALS_DIR})
endif()

if (DEFINED ENV{VISTA_CMAKE_CONFIG_DIR})
  SET(VistaCoreLibs_DIR "$ENV{VISTA_CMAKE_CONFIG_DIR}")
else()
  SET(VistaCoreLibs_DIR ${COSMOSCOUT_EXTERNALS_DIR}/share/VistaCoreLibs/cmake)
endif()

find_package(GLM REQUIRED)
find_package(GLI REQUIRED)
find_package(DOCTEST REQUIRED)
find_package(TINYGLTF REQUIRED)
find_package(CSPICE REQUIRED)
find_package(CARES REQUIRED)
find_package(CURL REQUIRED)
find_package(CURLPP REQUIRED)
find_package(SPDLOG REQUIRED)
find_package(CEF REQUIRED)
find_package(TIFF REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OPENSG REQUIRED)
find_package(CIVETWEB REQUIRED)
find_package(VistaCoreLibs REQUIRED 
  COMPONENTS "VistaBase" "VistaKernel" "VistaKernelOpenSGExt" "VistaOGLExt"
)

# X11 is used on Linux to set the application window's name and icon.
if (UNIX)
  find_package(X11)
endif()

# install some files -------------------------------------------------------------------------------

# Copy all third-party libraries to install directory.
install(
  DIRECTORY ${COSMOSCOUT_EXTERNALS_DIR}/lib/ ${COSMOSCOUT_EXTERNALS_DIR}/bin/ 
  DESTINATION "lib"
  FILES_MATCHING PATTERN "*.so*" PATTERN "*.dll*"
)

# Copy boost libraries to install directory.
foreach(BOOST_LIB ${Boost_LIBRARIES})
  if(EXISTS "${BOOST_LIB}")
    get_filename_component(LIB_BASE_NAME ${BOOST_LIB} NAME_WE)
    get_filename_component(LIB_PATH ${BOOST_LIB} PATH)
    if (WIN32)
      install(FILES ${LIB_PATH}/${LIB_BASE_NAME}.dll DESTINATION "lib")
    endif()
    if (UNIX)
      file(GLOB LIB_FILES ${LIB_PATH}/${LIB_BASE_NAME}.so*)
      install(FILES ${LIB_FILES} DESTINATION "lib")
    endif()
  endif()
endforeach()

# Install documentation directory
install(DIRECTORY
  ${CMAKE_SOURCE_DIR}/docs
  DESTINATION "."
)

# Install license files
install(FILES
  ${CMAKE_SOURCE_DIR}/LICENSE-3RD-PARTY.txt 
  ${CMAKE_SOURCE_DIR}/LICENSE.txt 
  DESTINATION "docs"
)

# create version header ----------------------------------------------------------------------------

message(STATUS "Trying to get current git branch and commit...")

# Get the current git branch name.
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get the current commit hash.
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Fallback to project version if it failed.
if (GIT_BRANCH STREQUAL "")
  message(STATUS "  Failed - falling back to project version: v${PROJECT_VERSION}")
else()
  message(STATUS "  Success: v${PROJECT_VERSION} (${GIT_BRANCH} @${GIT_COMMIT_HASH})")
endif()

configure_file(
  ${CMAKE_SOURCE_DIR}/src/cs-core/cs-version.hpp.in
  ${CMAKE_BINARY_DIR}/src/cs-core/cs-version.hpp
)

# compiler settings --------------------------------------------------------------------------------

add_definitions(
  -DBOOST_ALL_DYN_LINK
  -DGLM_ENABLE_EXPERIMENTAL
  -DGLM_FORCE_SWIZZLE
  -DNOMINMAX
  -DSPDLOG_COMPILED_LIB
)

if (MSVC)
  add_definitions(
    # For whatever reason ViSTA checks for 'WIN32' instead of '_WIN32' in a lot of places, which
    # results in Linux code being included on Windows.
    -DWIN32
  )

  add_compile_options(
    -experimental:external
    -external:anglebrackets
    -external:W0
    -W3 -WX
    -EHsc
    -wd4251

    # Warns about multiple inheritance problems, which we can't avoid inheriting from ViSTA classes.
    -wd4250
  )
else()
  add_compile_options(
    -Wall
    -Werror
  )
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable unit tests
option(COSMOSCOUT_UNIT_TESTS "Enable compilation of tests" OFF)
if (NOT COSMOSCOUT_UNIT_TESTS)
  add_definitions(-DDOCTEST_CONFIG_DISABLE)
endif()

# Enable code coverage measurements
option(COSMOSCOUT_COVERAGE_INFO "Run code coverage analytics" OFF)

if(COSMOSCOUT_COVERAGE_INFO AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  add_definitions(--coverage)
  add_link_options(--coverage)
endif()

# subdirectories -----------------------------------------------------------------------------------

include_directories(
  ${CMAKE_BINARY_DIR}/src/cs-core
  ${CMAKE_BINARY_DIR}/src/cs-utils
  ${CMAKE_BINARY_DIR}/src/cs-graphics
  ${CMAKE_BINARY_DIR}/src/cs-gui
  ${CMAKE_BINARY_DIR}/src/cs-scene
)

add_subdirectory(config)
add_subdirectory(resources)
add_subdirectory(src)
add_subdirectory(plugins)
