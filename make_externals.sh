#!/bin/bash

# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# exit on error
set -e

# ------------------------------------------------------------------------------------------------ #
# Make sure to run "git submodule update --init" before executing this script!                     #
# Default build mode is release, if "export COSMOSCOUT_DEBUG_BUILD=true" is executed before, all   #
# dependecies will be built in debug mode.                                                         #
# Usage:                                                                                           #
#    ./make_externals.sh [additional CMake flags, defaults to -G "Eclipse CDT4 - Unix Makefiles"]  #
# Examples:                                                                                        #
#    ./make_externals.sh                                                                           #
#    ./make_externals.sh -G "Unix Makefiles"                                                       #
#    ./make_externals.sh -G "Eclipse CDT4 - Unix Makefiles" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \ #
#                        -DCMAKE_C_COMPILER_LAUNCHER=ccache                                        #
# ------------------------------------------------------------------------------------------------ #

# Create some required variables. ------------------------------------------------------------------

# The CMake generator and other flags can be passed as parameters.
CMAKE_FLAGS=(-G "Eclipse CDT4 - Unix Makefiles")
if [ $# -ne 0 ]; then
  CMAKE_FLAGS=( "$@" )
fi

# Check if ComoScout VR debug build is enabled with "export COSMOSCOUT_DEBUG_BUILD=true".
BUILD_TYPE=release
case "$COSMOSCOUT_DEBUG_BUILD" in
  (true) echo "CosmoScout VR debug build is enabled!"; BUILD_TYPE=debug;
esac

# This directory should contain all submodules - they are assumed to reside in the subdirectory 
# "externals" next to this script.
EXTERNALS_DIR="$( cd "$( dirname "$0" )" && pwd )/externals"

# Get the current directory - this is the default location for the build and install directory.
CURRENT_DIR="$(pwd)"

# The build directory.
BUILD_DIR="$CURRENT_DIR/build/linux-externals-$BUILD_TYPE"

# The install directory.
INSTALL_DIR="$CURRENT_DIR/install/linux-externals-$BUILD_TYPE"

# Create some default installation directories.
cmake -E make_directory "$INSTALL_DIR/lib"
cmake -E make_directory "$INSTALL_DIR/share"
cmake -E make_directory "$INSTALL_DIR/bin"
cmake -E make_directory "$INSTALL_DIR/include"

# glew ---------------------------------------------------------------------------------------------

echo ""
echo "Downloading, building and installing GLEW ..."
echo ""

cmake -E make_directory "$BUILD_DIR/glew/extracted" && cd "$BUILD_DIR/glew"
wget -nc https://netcologne.dl.sourceforge.net/project/glew/glew/2.1.0/glew-2.1.0.tgz

cd "$BUILD_DIR/glew/extracted"
cmake -E tar xzf ../glew-2.1.0.tgz
cd ..

cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$BUILD_DIR/glew/extracted/glew-2.1.0/build/cmake"
cmake --build . --target install --parallel 8

# ViSTA expects glew library to be called libGLEW.so
case "$COSMOSCOUT_DEBUG_BUILD" in
  (true) cp $INSTALL_DIR/lib/libGLEWd.so $INSTALL_DIR/lib/libGLEW.so;;
esac

# freeglut -----------------------------------------------------------------------------------------

echo ""
echo "Building and installing freeglut ..."
echo ""

cmake -E make_directory "$BUILD_DIR/freeglut" && cd "$BUILD_DIR/freeglut"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_INSTALL_LIBDIR=lib -DFREEGLUT_BUILD_DEMOS=Off -DFREEGLUT_BUILD_STATIC_LIBS=Off \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/freeglut/freeglut/freeglut"
cmake --build . --target install --parallel 8

cmake -E copy_directory "$EXTERNALS_DIR/freeglut/freeglut/freeglut/include/GL" \
                        "$INSTALL_DIR/include/GL"

# c-ares -------------------------------------------------------------------------------------------

echo ""
echo "Building and installing c-ares ..."
echo ""

cmake -E make_directory "$BUILD_DIR/c-ares" && cd "$BUILD_DIR/c-ares"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/c-ares"
cmake --build . --target install --parallel 8

# curl ---------------------------------------------------------------------------------------------

echo ""
echo "Building and installing curl ..."
echo ""

cmake -E make_directory "$BUILD_DIR/curl" && cd "$BUILD_DIR/curl"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DBUILD_TESTING=OFF -DBUILD_CURL_EXE=OFF -DENABLE_ARES=ON \
      -DCARES_INCLUDE_DIR="$INSTALL_DIR/include" \
      -DCARES_LIBRARY="$INSTALL_DIR/lib/libcares.so" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      "$EXTERNALS_DIR/curl"
cmake --build . --target install --parallel 8

# curlpp -------------------------------------------------------------------------------------------

echo ""
echo "Building and installing curlpp ..."
echo ""

CURL_LIB="libcurl.so"
case "$COSMOSCOUT_DEBUG_BUILD" in
  (true) CURL_LIB="libcurl-d.so";;
esac

cmake -E make_directory "$BUILD_DIR/curlpp" && cd "$BUILD_DIR/curlpp"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCURL_INCLUDE_DIR="$INSTALL_DIR/include" \
      -DCURL_LIBRARY="$INSTALL_DIR/lib/$CURL_LIB" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/curlpp"
cmake --build . --target install --parallel 8

# libtiff ------------------------------------------------------------------------------------------

echo ""
echo "Building and installing libtiff ..."
echo ""

cmake -E make_directory "$BUILD_DIR/libtiff" && cd "$BUILD_DIR/libtiff"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_INSTALL_FULL_LIBDIR=lib \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/libtiff"
cmake --build . --target install --parallel 8

# spdlog -------------------------------------------------------------------------------------------

echo ""
echo "Installing spdlog ..."
echo ""

cmake -E copy_directory "$EXTERNALS_DIR/spdlog/include/spdlog" "$INSTALL_DIR/include/spdlog"

# doctest ------------------------------------------------------------------------------------------

echo ""
echo "Installing doctest ..."
echo ""

cmake -E copy_directory "$EXTERNALS_DIR/doctest/doctest" "$INSTALL_DIR/include/doctest"

# gli ----------------------------------------------------------------------------------------------

echo ""
echo "Installing gli ..."
echo ""

cmake -E copy_directory "$EXTERNALS_DIR/gli/gli" "$INSTALL_DIR/include/gli"

# glm ----------------------------------------------------------------------------------------------

echo ""
echo "Installing glm ..."
echo ""

cmake -E copy_directory "$EXTERNALS_DIR/glm/glm" "$INSTALL_DIR/include/glm"

# tinygltf -----------------------------------------------------------------------------------------

echo ""
echo "Installing tinygltf ..."
echo ""

cmake -E copy "$EXTERNALS_DIR/tinygltf/json.hpp"          "$INSTALL_DIR/include"
cmake -E copy "$EXTERNALS_DIR/tinygltf/stb_image.h"       "$INSTALL_DIR/include"
cmake -E copy "$EXTERNALS_DIR/tinygltf/stb_image_write.h" "$INSTALL_DIR/include"
cmake -E copy "$EXTERNALS_DIR/tinygltf/tiny_gltf.h"       "$INSTALL_DIR/include"

# opensg -------------------------------------------------------------------------------------------

echo ""
echo "Building and installing opensg-1.8 ..."
echo ""

cmake -E make_directory "$BUILD_DIR/opensg-1.8" && cd "$BUILD_DIR/opensg-1.8"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DGLUT_INCLUDE_DIR="$INSTALL_DIR/include" -DGLUT_LIBRARY="$INSTALL_DIR/lib/libglut.so" \
      -DOPENSG_BUILD_TESTS=Off -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/opensg-1.8"
cmake --build . --target install --parallel 8

# vista --------------------------------------------------------------------------------------------

echo ""
echo "Building and installing vista ..."
echo ""

cmake -E make_directory "$BUILD_DIR/vista" && cd "$BUILD_DIR/vista"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_CXX_FLAGS="-std=c++11" -DVISTADRIVERS_BUILD_3DCSPACENAVIGATOR=On \
      -DVISTADEMO_ENABLED=Off -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DOPENSG_ROOT_DIR="$INSTALL_DIR" \
      "$EXTERNALS_DIR/vista"
cmake --build . --target install --parallel 8

# cspice -------------------------------------------------------------------------------------------

echo ""
echo "Downloading and installing cspice ..."
echo ""

cmake -E make_directory "$BUILD_DIR/cspice/extracted" && cd "$BUILD_DIR/cspice"
wget -nc http://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z

cd "$BUILD_DIR/cspice/extracted"
cmake -E tar xzf ../cspice.tar.Z

cmake -E copy_directory "$BUILD_DIR/cspice/extracted/cspice/include" "$INSTALL_DIR/include/cspice"
cmake -E copy "$BUILD_DIR/cspice/extracted/cspice/lib/cspice.a" "$INSTALL_DIR/lib"

# cef ----------------------------------------------------------------------------------------------

echo ""
echo "Downloading, building and installing cef ..."
echo ""

CEF_DIR=cef_binary_79.1.36+g90301bd+chromium-79.0.3945.130_linux64_minimal

cmake -E make_directory "$BUILD_DIR/cef/extracted" && cd "$BUILD_DIR/cef"
wget -nc http://opensource.spotify.com/cefbuilds/cef_binary_79.1.36%2Bg90301bd%2Bchromium-79.0.3945.130_linux64_minimal.tar.bz2

cd "$BUILD_DIR/cef/extracted"
cmake -E tar xfj ../$CEF_DIR.tar.bz2
rm -rf $CEF_DIR/tests # we dont want the example applications
cd ..

cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCEF_COMPILER_FLAGS="-Wno-undefined-var-template" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$BUILD_DIR/cef/extracted/$CEF_DIR"
cmake --build . --parallel 8

cmake -E make_directory "$INSTALL_DIR/include/cef"
cmake -E copy_directory "$BUILD_DIR/cef/extracted/$CEF_DIR/include"    "$INSTALL_DIR/include/cef/include"
cmake -E copy_directory "$BUILD_DIR/cef/extracted/$CEF_DIR/Resources"  "$INSTALL_DIR/share/cef"
cmake -E copy_directory "$BUILD_DIR/cef/extracted/$CEF_DIR/Release"    "$INSTALL_DIR/lib"
cmake -E copy "$BUILD_DIR/cef/libcef_dll_wrapper/libcef_dll_wrapper.a" "$INSTALL_DIR/lib"

# --------------------------------------------------------------------------------------------------

if [ -e "$INSTALL_DIR/lib64" ]; then
      cmake -E copy_directory "$INSTALL_DIR/lib64" "$INSTALL_DIR/lib"
fi

echo "Finished successfully."
