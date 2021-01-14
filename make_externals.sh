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
# dependencies will be built in debug mode.                                                        #
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
BUILD_TYPE=Release
case "$COSMOSCOUT_DEBUG_BUILD" in
  (true) echo "CosmoScout VR debug build is enabled!"; BUILD_TYPE=Debug;
esac

# Check if unity build is disabled with "export COSMOSCOUT_USE_UNITY_BUILD=false".
UNITY_BUILD=On
case "$COSMOSCOUT_USE_UNITY_BUILD" in
  (false) echo "Unity build is disabled!"; UNITY_BUILD=Off;
esac

# Check if precompiled headers should not be used with "export COSMOSCOUT_USE_PCH=false".
PRECOMPILED_HEADERS=On
case "$COSMOSCOUT_USE_PCH" in
  (false) echo "Precompiled headers are disabled!"; PRECOMPILED_HEADERS=Off;
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

# TBB ------------------------------------------------------------------------------------------

echo ""
echo "Building and installing TBB V2019..."
echo ""

cmake -E make_directory "$BUILD_DIR/tbb/extracted" && cd "$BUILD_DIR/tbb"
wget -nc https://github.com/01org/tbb/releases/download/2019_U5/tbb2019_20190320oss_lin.tgz

cd "$BUILD_DIR/tbb/extracted"
cmake -E tar xzf ../tbb2019_20190320oss_lin.tgz

cmake -E copy_directory "$BUILD_DIR/tbb/extracted/tbb2019_20190320oss/include" "$INSTALL_DIR/include"
cmake -E copy_directory "$BUILD_DIR/tbb/extracted/tbb2019_20190320oss/lib"   	 "$INSTALL_DIR/lib"

# ispc -----------------------------------------------------------------------------------------

echo ""
echo "Downloading ispc..."
echo ""

cmake -E make_directory "$BUILD_DIR/ispc/extracted" && cd "$BUILD_DIR/ispc"
wget -nc https://github.com/ispc/ispc/releases/download/v1.14.1/ispc-v1.14.1-linux.tar.gz

cd "$BUILD_DIR/ispc/extracted"
cmake -E tar xzf ../ispc-v1.14.1-linux.tar.gz

cmake -E copy "$BUILD_DIR/ispc/extracted/ispc-v1.14.1-linux/bin/ispc" "$INSTALL_DIR/bin"

# rkcommon -----------------------------------------------------------------------------------------

echo ""
echo "Building and installing rkcommon..."
echo ""

cmake -E make_directory "$BUILD_DIR/rkcommon" && cd "$BUILD_DIR/rkcommon"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DINSTALL_DEPS=OFF -DBUILD_TESTING=OFF -DRKCOMMON_TBB_ROOT="$INSTALL_DIR" \
	"$EXTERNALS_DIR/rkcommon"
cmake --build . --config $BUILD_TYPE --target install --parallel "$(nproc)"

# embree -----------------------------------------------------------------------------------------

echo ""
echo "Building and installing embree..."
echo ""

cmake -E make_directory "$BUILD_DIR/embree" && cd "$BUILD_DIR/embree"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DEMBREE_TUTORIALS=OFF -DEMBREE_TBB_ROOT="$INSTALL_DIR" -DEMBREE_ISPC_EXECUTABLE="$INSTALL_DIR/bin/ispc" -DBUILD_TESTING=OFF \
	"$EXTERNALS_DIR/embree"
cmake --build . --config $BUILD_TYPE --target install --parallel "$(nproc)"

# openvkl -----------------------------------------------------------------------------------------

echo ""
echo "Building and installing openvkl..."
echo ""

cmake -E make_directory "$BUILD_DIR/openvkl" && cd "$BUILD_DIR/openvkl"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DRKCOMMON_TBB_ROOT="$INSTALL_DIR" -DISPC_EXECUTABLE="$INSTALL_DIR/bin/ispc" -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
	"$EXTERNALS_DIR/openvkl"
cmake --build . --config $BUILD_TYPE --target install --parallel "$(nproc)"

# oidn -----------------------------------------------------------------------------------------

echo ""
echo "Building and installing oidn..."
echo ""

cmake -E make_directory "$BUILD_DIR/oidn" && cd "$BUILD_DIR/oidn"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DTBB_ROOT="$INSTALL_DIR" -DOIDN_APPS=Off \
	"$EXTERNALS_DIR/oidn"
cmake --build . --config $BUILD_TYPE --target install --parallel "$(nproc)"

# Ospray -----------------------------------------------------------------------------------------

echo ""
echo "Building and installing ospray..."
echo ""

cmake -E make_directory "$BUILD_DIR/ospray" && cd "$BUILD_DIR/ospray"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
	-DRKCOMMON_TBB_ROOT="$INSTALL_DIR" -DISPC_EXECUTABLE="$INSTALL_DIR/bin/ispc" \
	-DOSPRAY_ENABLE_APPS=Off -DOSPRAY_MODULE_DENOISER=On -DOSPRAY_INSTALL_DEPENDENCIES=Off \
	-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=On \
	"$EXTERNALS_DIR/ospray"
cmake --build . --config $BUILD_TYPE --target install --parallel "$(nproc)"

# Zipper ---------------------------------------------------------------------------------------------

echo ""
echo "Building and installing zipper ..."
echo ""

cmake -E make_directory "$BUILD_DIR/zipper" && cd "$BUILD_DIR/zipper"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/zipper"
cmake --build . --target install --parallel "$(nproc)"

# Proj6 ---------------------------------------------------------------------------------------------

echo ""
echo "Downloading, building and installing PROJ6 ..."
echo ""

# SQLITE Binary
cd "$BUILD_DIR"
wget -nc https://github.com/boramalper/sqlite3-x64/releases/download/3310100--2020-02-18T12.16.42Z/sqlite3
chmod +x "$BUILD_DIR/sqlite3"

cmake -E make_directory "$BUILD_DIR/proj6/extracted" && cd "$BUILD_DIR/proj6"
wget -nc https://download.osgeo.org/proj/proj-6.3.2.tar.gz

cd "$BUILD_DIR/proj6/extracted"
cmake -E tar xzf ../proj-6.3.2.tar.gz
cd "$BUILD_DIR/proj6/extracted/proj-6.3.2"

cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DPROJ_TESTS=OFF \
      -DEXE_SQLITE3="$BUILD_DIR/sqlite3" \
      -DSQLITE3_INCLUDE_DIR="$EXTERNALS_DIR/sqlite3" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" "$BUILD_DIR/proj6/extracted/proj-6.3.2"
cmake --build . --target install --parallel "$(nproc)"

# gdal 3.2.0 ----------------------------------------------------------------------------------------

echo ""
echo "Downloading and installing gdal ..."
echo ""

cmake -E make_directory "$BUILD_DIR/gdal/extracted" && cd "$BUILD_DIR/gdal"
wget -nc https://github.com/OSGeo/gdal/releases/download/v3.2.0/gdal-3.2.0.tar.gz

cd "$BUILD_DIR/gdal/extracted"
cmake -E tar xzf ../gdal-3.2.0.tar.gz
cd "$BUILD_DIR/gdal/extracted/gdal-3.2.0"

./configure --prefix="$INSTALL_DIR" \
  --with-proj="$INSTALL_DIR"

make -j"$(nproc)"
make install

# VTK -----------------------------------------------------------------------------------------

echo ""
echo "Building and installing VTK 9.0.1 ..."
echo ""

echo ""
echo "Patching VTK ..."
echo ""

cd $EXTERNALS_DIR/vtk/IO
cmake -E tar xfvj $EXTERNALS_DIR/../VTK-Patch.zip

cmake -E make_directory $BUILD_DIR/vtk && cd $BUILD_DIR/vtk
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DVTK_BUILD_TESTING=OFF -DVTK_BUILD_EXAMPLES=OFF \
			-DVTK_GROUP_ENABLE_Imaging=DONT_WANT \
			-DVTK_GROUP_ENABLE_MPI=DONT_WANT \
			-DVTK_GROUP_ENABLE_Qt=DONT_WANT \
			-DVTK_GROUP_ENABLE_Rendering=DONT_WANT \
			-DVTK_GROUP_ENABLE_StandAlone=DONT_WANT \
			-DVTK_GROUP_ENABLE_Views=DONT_WANT \
			-DVTK_GROUP_ENABLE_Web=DONT_WANT \
			-DVTK_MODULE_ENABLE_VTK_CommonCore=YES \
			-DVTK_MODULE_ENABLE_VTK_CommonDataModel=YES \
			-DVTK_MODULE_ENABLE_VTK_FiltersCore=YES \
			-DVTK_MODULE_ENABLE_VTK_FiltersGeometry=YES \
			-DVTK_MODULE_ENABLE_VTK_IOInfovis=YES \
			-DVTK_MODULE_ENABLE_VTK_IOImage=YES \
			-DVTK_MODULE_ENABLE_VTK_IOLegacy=YES \
			-DVTK_MODULE_ENABLE_VTK_IOXML=YES \
			-DVTK_MODULE_ENABLE_VTK_IOWeb=YES \
			$EXTERNALS_DIR/vtk
cmake --build . --config $BUILD_TYPE --target install --parallel 8

# TTK -----------------------------------------------------------------------------------------

echo ""
echo "Building and installing TTK 0.9.9 ..."
echo ""

REQUIRED=" \
			-DVTK_MODULE_ENABLE_ttkCinemaProductReader=YES \
			-DVTK_MODULE_ENABLE_ttkCinemaQuery=YES \
			-DVTK_MODULE_ENABLE_ttkCinemaReader=YES"
DEPENDENCIES=" \
			-DVTK_MODULE_ENABLE_ttkAlgorithm=YES \
			-DVTK_MODULE_ENABLE_ttkTopologicalCompressionReader=YES"

cmake -E make_directory $BUILD_DIR/ttk && cd $BUILD_DIR/ttk
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DTTK_BUILD_PARAVIEW_PLUGINS=Off -DTTK_BUILD_STANDALONE_APPS=Off -DBUILD_TESTING=off \
			-DTTK_ENABLE_GRAPHVIZ=Off -DTTK_ENABLE_EIGEN=Off -DTTK_ENABLE_EMBREE=Off -DTTK_WHITELIST_MODE=On \
			$REQUIRED $DEPENDENCIES \
			-DVTK_MODULE_ENABLE_ttkWRLExporter=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkUserInterfaceBase=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkUncertainDataEstimator=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTriangulationRequest=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTriangulationAlgorithm=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTrackingFromOverlap=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTrackingFromFields=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTrackingFromPersistenceDiagrams=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTopologicalSimplification=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTopologicalCompression=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTextureMapFromField=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTableDataSelector=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkStringArrayConverter=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkSphereFromPoint=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkScalarFieldSmoother=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkScalarFieldNormalizer=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkScalarFieldCriticalPoints=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkReebSpace=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkRangePolygon=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkQuadrangulationSubdivision=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkProjectionFromField=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkProgramBase=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPointSetToCurve=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPointMerger=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPointDataSelector=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPointDataConverter=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPlanarGraphLayout=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPersistenceDiagramDistanceMatrix=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPersistenceDiagramClustering=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPersistenceDiagram=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPersistenceCurve=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkPeriodicGrid=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkOFFWriter=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkOFFReader=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkOBJWriter=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkMorseSmaleQuadrangulation=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkMorseSmaleComplex=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkMeshSubdivision=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkMeshGraph=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkMatrixToHeatMap=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkManifoldCheck=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkMandatoryCriticalPoints=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkLDistanceMatrix=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkLDistance=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkJacobiSet=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkIntegralLines=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkImportEmbeddingFromTable=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkIdentifyByScalarField=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkIdentifiers=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkIdentifierRandomizer=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkIcospheresFromPoints=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkIcosphereFromObject=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkIcosphere=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkHelloWorld=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkHarmonicField=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkGridLayout=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkGeometrySmoother=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkGaussianPointCloud=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkFiberSurface=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkFiber=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkFTRGraph=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkFTMTree=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkEndFor=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkForEach=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkExtract=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkEigenField=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkDistanceField=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkDiscreteGradient=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkDimensionReduction=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkDepthImageBasedGeometryApproximation=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkDataSetToTable=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkDataSetInterpolator=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkContourTreeAlignment=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkContourForests=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkContourAroundPoint=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkContinuousScatterPlot=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkComponentSize=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkCinemaWriter=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkTopologicalCompressionWriter=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkCinemaImaging=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkCinemaDarkroom=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkBottleneckDistance=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkBlockAggregator=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkBlank=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkBarycentricSubdivision=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkArrayPreconditioning=DONT_WANT \
			-DVTK_MODULE_ENABLE_ttkArrayEditor=DONT_WANT \
      $EXTERNALS_DIR/ttk
cmake --build . --config $BUILD_TYPE --target install --parallel 8

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
cmake --build . --target install --parallel "$(nproc)"

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
cmake --build . --target install --parallel "$(nproc)"

cmake -E copy_directory "$EXTERNALS_DIR/freeglut/freeglut/freeglut/include/GL" \
                        "$INSTALL_DIR/include/GL"

# c-ares -------------------------------------------------------------------------------------------

echo ""
echo "Building and installing c-ares ..."
echo ""

cmake -E make_directory "$BUILD_DIR/c-ares" && cd "$BUILD_DIR/c-ares"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/c-ares"
cmake --build . --target install --parallel "$(nproc)"

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
cmake --build . --target install --parallel "$(nproc)"

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
      -DCURL_INCLUDE_DIR="$INSTALL_DIR/include" -DCMAKE_UNITY_BUILD=$UNITY_BUILD \
      -DCURL_LIBRARY="$INSTALL_DIR/lib/$CURL_LIB" \
      -DCMAKE_INSTALL_LIBDIR=lib -DCURL_NO_CURL_CMAKE=On \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/curlpp"
cmake --build . --target install --parallel "$(nproc)"

# libtiff ------------------------------------------------------------------------------------------

echo ""
echo "Building and installing libtiff ..."
echo ""

cmake -E make_directory "$BUILD_DIR/libtiff" && cd "$BUILD_DIR/libtiff"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_INSTALL_FULL_LIBDIR=lib -DCMAKE_UNITY_BUILD=$UNITY_BUILD \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/libtiff"
cmake --build . --target install --parallel "$(nproc)"

# spdlog -------------------------------------------------------------------------------------------

echo ""
echo "Building and installing spdlog ..."
echo ""

cmake -E make_directory "$BUILD_DIR/spdlog" && cd "$BUILD_DIR/spdlog"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_POSITION_INDEPENDENT_CODE=On -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/spdlog"
cmake --build . --target install --parallel "$(nproc)"

# civetweb -----------------------------------------------------------------------------------------

echo ""
echo "Building and installing civetweb ..."
echo ""

cmake -E make_directory "$BUILD_DIR/civetweb" && cd "$BUILD_DIR/civetweb"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCIVETWEB_ENABLE_CXX=On \
      -DCIVETWEB_BUILD_TESTING=Off \
      -DBUILD_SHARED_LIBS=On -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/civetweb"
cmake --build . --target install --parallel "$(nproc)"

# jsonhpp ------------------------------------------------------------------------------------------

echo ""
echo "Installing jsonHPP ..."
echo ""

cmake -E copy_directory "$EXTERNALS_DIR/json/include/nlohmann" "$INSTALL_DIR/include/nlohmann"

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

cmake -E copy "$EXTERNALS_DIR/tinygltf/json.hpp"    "$INSTALL_DIR/include"
cmake -E copy "$EXTERNALS_DIR/tinygltf/tiny_gltf.h" "$INSTALL_DIR/include"

# stb ----------------------------------------------------------------------------------------------

echo ""
echo "Installing stb ..."
echo ""

cmake -E copy "$EXTERNALS_DIR/stb/stb_image.h"        "$INSTALL_DIR/include"
cmake -E copy "$EXTERNALS_DIR/stb/stb_image_write.h"  "$INSTALL_DIR/include"
cmake -E copy "$EXTERNALS_DIR/stb/stb_image_resize.h" "$INSTALL_DIR/include"

# opensg -------------------------------------------------------------------------------------------

echo ""
echo "Building and installing opensg-1.8 ..."
echo ""

cmake -E make_directory "$BUILD_DIR/opensg-1.8" && cd "$BUILD_DIR/opensg-1.8"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_UNITY_BUILD=$UNITY_BUILD \
      -DOPENSG_USE_PRECOMPILED_HEADERS=$PRECOMPILED_HEADERS \
      -DGLUT_INCLUDE_DIR="$INSTALL_DIR/include" -DGLUT_LIBRARY="$INSTALL_DIR/lib/libglut.so" \
      -DOPENSG_BUILD_TESTS=Off -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$EXTERNALS_DIR/opensg-1.8"
cmake --build . --target install --parallel "$(nproc)"

# vista --------------------------------------------------------------------------------------------

echo ""
echo "Building and installing vista ..."
echo ""

cmake -E make_directory "$BUILD_DIR/vista" && cd "$BUILD_DIR/vista"
cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" -DCMAKE_UNITY_BUILD=$UNITY_BUILD \
      -DVISTA_USE_PRECOMPILED_HEADERS=$PRECOMPILED_HEADERS \
      -DCMAKE_CXX_FLAGS="-std=c++11" -DVISTADRIVERS_BUILD_3DCSPACENAVIGATOR=On \
      -DVISTADEMO_ENABLED=Off -DCMAKE_BUILD_TYPE=$BUILD_TYPE -DOPENSG_ROOT_DIR="$INSTALL_DIR" \
      "$EXTERNALS_DIR/vista"
cmake --build . --target install --parallel "$(nproc)"

# cspice -------------------------------------------------------------------------------------------

echo ""
echo "Downloading and installing cspice ..."
echo ""

cmake -E make_directory "$BUILD_DIR/cspice/extracted" && cd "$BUILD_DIR/cspice"
wget -nc http://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Linux_GCC_64bit/packages/cspice.tar.Z

cd "$BUILD_DIR/cspice/extracted"
cmake -E tar xzf ../cspice.tar.Z -- cspice/lib/cspice.a cspice/include

cmake -E copy_directory "$BUILD_DIR/cspice/extracted/cspice/include" "$INSTALL_DIR/include/cspice"
cmake -E copy "$BUILD_DIR/cspice/extracted/cspice/lib/cspice.a" "$INSTALL_DIR/lib"

# cef ----------------------------------------------------------------------------------------------

echo ""
echo "Downloading, building and installing cef ..."
echo ""

CEF_DIR=cef_binary_88.1.6+g4fe33a1+chromium-88.0.4324.96_linux64_minimal

cmake -E make_directory "$BUILD_DIR/cef/extracted" && cd "$BUILD_DIR/cef"
wget -nc https://cef-builds.spotifycdn.com/cef_binary_88.1.6%2Bg4fe33a1%2Bchromium-88.0.4324.96_linux64_minimal.tar.bz2

cd "$BUILD_DIR/cef/extracted"
cmake -E tar xfj ../$CEF_DIR.tar.bz2
rm -rf $CEF_DIR/tests # we dont want the example applications
cd ..

cmake "${CMAKE_FLAGS[@]}" -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCEF_COMPILER_FLAGS="-Wno-undefined-var-template" -DCMAKE_UNITY_BUILD=$UNITY_BUILD \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE "$BUILD_DIR/cef/extracted/$CEF_DIR"
cmake --build . --parallel "$(nproc)"

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
