@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                              This file is part of CosmoScout VR                                #
rem     and may be used under the terms of the MIT license. See the LICENSE file for details.      #
rem                       Copyright: (c) 2019 German Aerospace Center (DLR)                        #
rem ---------------------------------------------------------------------------------------------- #

rem ---------------------------------------------------------------------------------------------- #
rem usage: make_externals.bat [path_to_build_directory] [path_to_install_directory]                #
rem Make sure to run "git submodule update --init" before executing this script!                   #
rem ---------------------------------------------------------------------------------------------- #

rem Create some required variables. ----------------------------------------------------------------

rem This directory should all the submodules - they are assumed to reside in the subdirectory 
rem "externals" next to this script.
set EXTERNALS_DIR=%~dp0\externals

rem Get the current directory - this is the default location for the build and install directory.
set CURRENT_DIR=%cd%

rem The build directory can be passed as first parameter.
set BUILD_DIR=%CURRENT_DIR%\build\windows-externals

rem The install directory can be passed as second parameter.
set INSTALL_DIR=%CURRENT_DIR%\install\windows-externals

rem Create some default installation directories.
cmake -E make_directory "%INSTALL_DIR%/lib"
cmake -E make_directory "%INSTALL_DIR%/share"
cmake -E make_directory "%INSTALL_DIR%/bin"
cmake -E make_directory "%INSTALL_DIR%/include"

rem glew -------------------------------------------------------------------------------------------

echo.
echo Downloading, building and installing GLEW ...
echo.

cmake -E make_directory "%BUILD_DIR%/glew/extracted" && cd "%BUILD_DIR%/glew"
powershell.exe -command Invoke-WebRequest -Uri https://netix.dl.sourceforge.net/project/glew/glew/2.1.0/glew-2.1.0.zip -OutFile glew-2.1.0.zip

cd "%BUILD_DIR%/glew/extracted"
cmake -E tar xfvj ../glew-2.1.0.zip
cd ..

cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_INSTALL_LIBDIR=lib^
      "%BUILD_DIR%/glew/extracted/glew-2.1.0/build/cmake" || exit /b
cmake --build . --config Release --target install --parallel 8 || exit /b

rem freeglut ---------------------------------------------------------------------------------------

echo.
echo Building and installing freeglut ...
echo.

cmake -E make_directory "%BUILD_DIR%/freeglut" && cd "%BUILD_DIR%/freeglut"
cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_INSTALL_LIBDIR=lib^
      "%EXTERNALS_DIR%/freeglut/freeglut/freeglut" || exit /b
cmake --build . --config Release --target install --parallel 8 || exit /b

cmake -E copy_directory "%EXTERNALS_DIR%/freeglut/freeglut/freeglut/include/GL" "%INSTALL_DIR%/include/GL"

rem c-ares -----------------------------------------------------------------------------------------

echo.
echo Building and installing c-ares ...
echo.

cmake -E make_directory "%BUILD_DIR%/c-ares" && cd "%BUILD_DIR%/c-ares"
cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      "%EXTERNALS_DIR%/c-ares" || exit /b
cmake --build . --config Release --target install --parallel 8 || exit /b

rem curl -------------------------------------------------------------------------------------------

echo.
echo Building and installing curl ...
echo.

cmake -E make_directory "%BUILD_DIR%/curl" && cd "%BUILD_DIR%/curl"
cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DBUILD_TESTING=OFF -DBUILD_CURL_EXE=OFF -DENABLE_ARES=ON^
      -DCARES_INCLUDE_DIR="%INSTALL_DIR%/include"^
      -DCARES_LIBRARY="%INSTALL_DIR%/lib/cares.lib"^
      -DCMAKE_INSTALL_LIBDIR=lib^
      "%EXTERNALS_DIR%/curl" || exit /b
cmake --build . --config Release --target install --parallel 8 || exit /b

rem curlpp -----------------------------------------------------------------------------------------

echo.
echo Building and installing curlpp ...
echo.

cmake -E make_directory "%BUILD_DIR%/curlpp" && cd "%BUILD_DIR%/curlpp"
cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCURL_INCLUDE_DIR="%INSTALL_DIR%/include"^
      -DCURL_LIBRARY="%INSTALL_DIR%/lib/libcurl_imp.lib"^
      -DCMAKE_INSTALL_LIBDIR=lib^
      "%EXTERNALS_DIR%/curlpp" || exit /b
cmake --build . --config Release --target install --parallel 8 || exit /b

rem libtiff ----------------------------------------------------------------------------------------

echo.
echo Building and installing libtiff ...
echo.

cmake -E make_directory "%BUILD_DIR%/libtiff" && cd "%BUILD_DIR%/libtiff"
cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_INSTALL_FULL_LIBDIR=lib^
      "%EXTERNALS_DIR%/libtiff" || exit /b
cmake --build . --config Release --target install --parallel 8 || exit /b

rem gli --------------------------------------------------------------------------------------------

echo.
echo Installing gli ...
echo.

cmake -E copy_directory "%EXTERNALS_DIR%/gli/gli" "%INSTALL_DIR%/include/gli" || exit /b

rem glm --------------------------------------------------------------------------------------------

echo.
echo Installing glm ...
echo.

cmake -E copy_directory "%EXTERNALS_DIR%/glm/glm" "%INSTALL_DIR%/include/glm" || exit /b

rem tinygltf ---------------------------------------------------------------------------------------

echo.
echo Installing tinygltf ...
echo.

cmake -E copy "%EXTERNALS_DIR%/tinygltf/json.hpp"          "%INSTALL_DIR%/include" || exit /b
cmake -E copy "%EXTERNALS_DIR%/tinygltf/stb_image.h"       "%INSTALL_DIR%/include" || exit /b
cmake -E copy "%EXTERNALS_DIR%/tinygltf/stb_image_write.h" "%INSTALL_DIR%/include" || exit /b
cmake -E copy "%EXTERNALS_DIR%/tinygltf/tiny_gltf.h"       "%INSTALL_DIR%/include" || exit /b

rem opensg -----------------------------------------------------------------------------------------

echo.
echo Building and installing opensg-1.8 ...
echo.

cmake -E make_directory "%BUILD_DIR%/opensg-1.8" && cd "%BUILD_DIR%/opensg-1.8"
cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DGLUT_INCLUDE_DIR="%INSTALL_DIR%/include" -DGLUT_LIBRARY="%INSTALL_DIR%/lib/freeglut.lib"^
      "%EXTERNALS_DIR%/opensg-1.8"
cmake --build . --config Release --target install --parallel 8 || exit /b

rem vista ------------------------------------------------------------------------------------------

echo.
echo Building and installing vista ...
echo.

cmake -E make_directory "%BUILD_DIR%/vista" && cd "%BUILD_DIR%/vista"

rem set OPENVR="T:/modulesystem/tools/openvr/OpenVR_SDK_1.0.3/install/win7.x86_64.msvc14.release"
rem cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem       -DVISTACORELIBS_USE_VIVE=On -DVISTADRIVERS_BUILD_VIVE=On -DOPENVR_ROOT_DIR=%OPENVR%^
rem       -DVISTADRIVERS_BUILD_3DCSPACENAVIGATOR=On^
rem       -DCMAKE_CXX_FLAGS="-std=c++11" "%EXTERNALS_DIR%/vista" || exit /b

cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_CXX_FLAGS="-std=c++11" "%EXTERNALS_DIR%/vista" || exit /b

cmake --build . --config Release --target install --parallel 8 || exit /b

rem cspice -----------------------------------------------------------------------------------------

echo.
echo Downloading and installing cspice ...
echo.

cmake -E make_directory "%BUILD_DIR%/cspice/extracted" && cd "%BUILD_DIR%/cspice"
powershell.exe -command $AllProtocols = [System.Net.SecurityProtocolType]'Tls11,Tls12'; [System.Net.ServicePointManager]::SecurityProtocol = $AllProtocols; Invoke-WebRequest -Uri https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Windows_VisualC_64bit/packages/cspice.zip -OutFile cspice.zip

cd "%BUILD_DIR%/cspice/extracted"
cmake -E tar xfvj ../cspice.zip
cd cspice

echo project(cspice C) > "CMakeLists.txt"
echo cmake_minimum_required(VERSION 2.8) >> "CMakeLists.txt"
echo add_definitions("-D_COMPLEX_DEFINED -DMSDOS -DOMIT_BLANK_CC -DKR_headers -DNON_ANSI_STDIO") >> "CMakeLists.txt"
echo file(GLOB_RECURSE CSPICE_SOURCE src/cspice/*.c) >> "CMakeLists.txt"
echo add_library(cspice SHARED ${CSPICE_SOURCE}) >> "CMakeLists.txt"
echo set_target_properties(cspice PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1) >> "CMakeLists.txt"

cmake -G "Visual Studio 15 Win64" . || exit /b
cmake --build . --config Release --parallel 8 || exit /b

cmake -E copy_directory "%BUILD_DIR%/cspice/extracted/cspice/include"            "%INSTALL_DIR%/include/cspice"
cmake -E copy           "%BUILD_DIR%/cspice/extracted/cspice/Release/cspice.lib" "%INSTALL_DIR%/lib"

rem cef --------------------------------------------------------------------------------------------

echo.
echo Downloading bzip2 ...
echo.

cmake -E make_directory "%BUILD_DIR%/cef/bzip2" && cd "%BUILD_DIR%/cef"
powershell.exe -command Invoke-WebRequest -Uri https://kent.dl.sourceforge.net/project/gnuwin32/bzip2/1.0.5/bzip2-1.0.5-bin.zip -OutFile bzip2.zip
cd "%BUILD_DIR%/cef/bzip2"
cmake -E tar xfvj ../bzip2.zip
cd ..

echo.
echo Downloading, building and installing cef (this may take some time) ...
echo.

set CEF_VERSION=cef_binary_3.3239.1723.g071d1c1_windows64

cmake -E make_directory "%BUILD_DIR%/cef/extracted" && cd "%BUILD_DIR%/cef"
powershell.exe -command Invoke-WebRequest -Uri http://opensource.spotify.com/cefbuilds/%CEF_VERSION%.tar.bz2 -OutFile cef.tar.bz2

cd "%BUILD_DIR%/cef/extracted"
"%BUILD_DIR%/cef/bzip2/bin/bunzip2.exe" -v ../cef.tar.bz2
cmake -E tar xfvj ../cef.tar

rem We dont want the example applications.
rmdir %CEF_VERSION%\tests /s /q

rem We wnat to built with /MD
powershell.exe -command "(gc %CEF_VERSION%\cmake\cef_variables.cmake) -replace '/MT', '/MD' | Out-File -encoding UTF8 %CEF_VERSION%\cmake\cef_variables.cmake"

cd ..

cmake -G "Visual Studio 15 Win64" -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      "%BUILD_DIR%/cef/extracted/%CEF_VERSION%" || exit /b
cmake --build . --config Release --parallel 8 || exit /b

echo Installing cef...
cmake -E make_directory "%INSTALL_DIR%/include/cef"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_VERSION%/include"         "%INSTALL_DIR%/include/cef/include"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_VERSION%/Resources"       "%INSTALL_DIR%/share/cef"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_VERSION%/Release"         "%INSTALL_DIR%/lib"
cmake -E copy "%BUILD_DIR%/cef/libcef_dll_wrapper/Release/libcef_dll_wrapper.lib" "%INSTALL_DIR%/lib"

rem ------------------------------------------------------------------------------------------------

cd "%CURRENT_DIR%"
echo Finished successfully.

@echo on
