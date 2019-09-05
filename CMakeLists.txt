# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

cmake_minimum_required(VERSION 3.12)
project(cosmoscout-vr VERSION 1.0.0)

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

if (DEFINED ENV{VISTA_CMAKE_CONFIG_DIR})
  SET(VistaCoreLibs_DIR "$ENV{VISTA_CMAKE_CONFIG_DIR}")
else()
  SET(VistaCoreLibs_DIR ${COSMOSCOUT_EXTERNALS_DIR}/share/VistaCoreLibs/cmake)
endif()

find_package(GLM REQUIRED)
find_package(GLI REQUIRED)
find_package(TINYGLTF REQUIRED)
find_package(CSPICE REQUIRED)
find_package(CARES REQUIRED)
find_package(CURL REQUIRED)
find_package(CURLPP REQUIRED)
find_package(CEF REQUIRED)
find_package(TIFF REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OPENSG REQUIRED)
find_package(VistaCoreLibs REQUIRED 
  COMPONENTS "VistaBase" "VistaKernel" "VistaKernelOpenSGExt" "VistaOGLExt"
)

# strip "optimzed" flags as we will be using release libs only.
list(FILTER VISTACORELIBS_LIBRARIES EXCLUDE REGEX optimized)

# copy all third-party libraries to install directory
install(
  DIRECTORY ${COSMOSCOUT_EXTERNALS_DIR}/lib/ ${COSMOSCOUT_EXTERNALS_DIR}/bin/ 
  DESTINATION "lib"
  FILES_MATCHING PATTERN "*.so*" PATTERN "*.dll*"
)

# copy boost libraries to install directory
if (WIN32)
  foreach(BOOST_LIB ${Boost_LIBRARIES})
    if(EXISTS "${BOOST_LIB}")
      get_filename_component(LIB_BASE_NAME ${BOOST_LIB} NAME_WE)
      get_filename_component(LIB_PATH ${BOOST_LIB} PATH)
      install(FILES ${LIB_PATH}/${LIB_BASE_NAME}.dll DESTINATION "lib")
    endif()
  endforeach()
endif()
if (UNIX)
  install(FILES ${Boost_LIBRARIES} DESTINATION "lib")
endif()

# create version header ----------------------------------------------------------------------------

message(STATUS "Trying to get current git branch and commit...")

# get the current git branch name
execute_process(
  COMMAND git rev-parse --abbrev-ref HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_BRANCH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# get the current commit hash
execute_process(
  COMMAND git log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# get the most recent tag
execute_process(
  COMMAND git describe --abbrev=0 --tags
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_RECENT_TAG
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# fallback to project version if it failed
if (GIT_BRANCH STREQUAL "")
  set(GIT_RECENT_TAG "v${PROJECT_VERSION}")
  message(STATUS "  Failed - falling back to project version: ${GIT_RECENT_TAG}")
else()
  message(STATUS "  Success: ${GIT_RECENT_TAG} (${GIT_BRANCH} @${GIT_COMMIT_HASH})")
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
)

add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/EHsc>)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# install data directory, complain if non-existent -------------------------------------------------

if (EXISTS ${CMAKE_SOURCE_DIR}/data)
  install(DIRECTORY ${CMAKE_SOURCE_DIR}/data DESTINATION "share/resources")
else()
  message(WARNING "Data directory was not found. Maybe you forgot to run the get_example_data.[sh/bat] script?")
endif()