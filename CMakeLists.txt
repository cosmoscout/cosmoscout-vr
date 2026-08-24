# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.28)
project(cosmoscout-vr VERSION 1.10.0)

# Ensure local modules (for dependencies etc.) are found.
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# CMAKE_BUILD_TYPE must be set (except for Visual Studio).
if(NOT MSVC)
  if(NOT CMAKE_BUILD_TYPE OR
    (NOT ${CMAKE_BUILD_TYPE} STREQUAL "Release" AND
    NOT ${CMAKE_BUILD_TYPE} STREQUAL "Debug"))

    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Release or Debug" FORCE)
  endif()


  set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
  set(CMAKE_SKIP_RPATH OFF)
  set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)
endif()

# Use folders when targeting an IDE
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# find dependencies --------------------------------------------------------------------------------

include(GenerateExportHeader)

find_package(OpenGL REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(gli CONFIG REQUIRED)
find_package(unofficial-cspice CONFIG REQUIRED)
find_package(CURL REQUIRED)
find_package(curlcpp CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(cef CONFIG REQUIRED)
find_package(TIFF REQUIRED COMPONENTS CXX)
find_package(GLEW CONFIG REQUIRED)
find_package(SDL2 CONFIG REQUIRED)
find_package(SDL2_ttf CONFIG REQUIRED)
find_package(OpenSG CONFIG REQUIRED)
find_package(Vista CONFIG REQUIRED)

# X11 is used on Linux to set the application window's name and icon.
if (UNIX)
  find_package(X11)
endif()

# install some files -------------------------------------------------------------------------------

# Copy all third-party libraries to install directory.
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  install(
      DIRECTORY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/lib/ ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/bin/
      DESTINATION "lib"
      FILES_MATCHING PATTERN "*.so*" PATTERN "*.dll*"
  )
else ()
  install(
      DIRECTORY ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/lib/ ${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin/
      DESTINATION "lib"
      FILES_MATCHING PATTERN "*.so*" PATTERN "*.dll*"
  )
endif ()

# Install documentation directory
install(DIRECTORY
  ${CMAKE_SOURCE_DIR}/docs
  DESTINATION "."
)

# Install license files
install(FILES
  ${CMAKE_SOURCE_DIR}/LICENSE-3RD-PARTY.txt
  ${CMAKE_SOURCE_DIR}/LICENSE.md
  DESTINATION "docs"
)

install(DIRECTORY
  ${CMAKE_SOURCE_DIR}/LICENSES
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
    $<$<COMPILE_LANGUAGE:CXX>:-experimental:external>
    $<$<COMPILE_LANGUAGE:CXX>:-external:anglebrackets>
    $<$<COMPILE_LANGUAGE:CXX>:-external:W0>
    $<$<COMPILE_LANGUAGE:CXX>:-W3>
    $<$<COMPILE_LANGUAGE:CXX>:-WX>
    $<$<COMPILE_LANGUAGE:CXX>:-EHsc>
    $<$<COMPILE_LANGUAGE:CXX>:-wd4251>

    # Warns about multiple inheritance problems, which we can't avoid inheriting from ViSTA classes.
    $<$<COMPILE_LANGUAGE:CXX>:-wd4250>
  )
else()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Werror>
  )
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable unit tests
option(COSMOSCOUT_UNIT_TESTS "Enable compilation of tests" OFF)
if (NOT COSMOSCOUT_UNIT_TESTS)
  add_definitions(-DDOCTEST_CONFIG_DISABLE)
else ()
  find_package(doctest CONFIG REQUIRED)
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
add_subdirectory(tools/eclipse-shadow-generator)
