@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                              This file is part of CosmoScout VR                                #
rem     and may be used under the terms of the MIT license. See the LICENSE file for details.      #
rem                       Copyright: (c) 2019 German Aerospace Center (DLR)                        #
rem ---------------------------------------------------------------------------------------------- #

rem ---------------------------------------------------------------------------------------------- #
rem Make sure to run "git submodule update --init" before executing this script!                   #
rem Usage:                                                                                         #
rem    make_externals.bat [additional CMake flags, defaults to -G "Visual Studio 15 Win64"]        #
rem Examples:                                                                                      #
rem    make_externals.bat                                                                          #
rem    make_externals.bat -G "Visual Studio 15 Win64"                                              #
rem    make_externals.bat -G "Visual Studio 16 2019" -A x64                                        #
rem ---------------------------------------------------------------------------------------------- #

rem The CMake generator and other flags can be passed as parameters.
set CMAKE_FLAGS=-G "Visual Studio 15 Win64"
IF NOT "%~1"=="" (
  SET CMAKE_FLAGS=%*
)

rem Check if ComoScout VR debug build is set with the environment variable
IF "%COSMOSCOUT_DEBUG_BUILD%"=="true" (
  ECHO CosmoScout VR debug build is enabled!
  set BUILD_TYPE=debug
) else (
  set BUILD_TYPE=release
)
rem Create some required variables. ----------------------------------------------------------------

rem This directory should contain all submodules - they are assumed to reside in the subdirectory 
rem "externals" next to this script.
set EXTERNALS_DIR=%~dp0\externals

rem Get the current directory - this is the default location for the build and install directory.
set CURRENT_DIR=%cd%

rem The build directory.
set BUILD_DIR=%CURRENT_DIR%\build\windows-externals-%BUILD_TYPE%

rem The install directory.
set INSTALL_DIR=%CURRENT_DIR%\install\windows-externals-%BUILD_TYPE%

rem Create some default installation directories.
cmake -E make_directory "%INSTALL_DIR%/lib"
cmake -E make_directory "%INSTALL_DIR%/share"
cmake -E make_directory "%INSTALL_DIR%/bin"
cmake -E make_directory "%INSTALL_DIR%/include"

rem rem glew -------------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Downloading, building and installing GLEW ...
rem echo.
rem 
rem cmake -E make_directory "%BUILD_DIR%/glew/extracted" && cd "%BUILD_DIR%/glew"
rem powershell.exe -command Invoke-WebRequest -Uri https://netcologne.dl.sourceforge.net/project/glew/glew/2.1.0/glew-2.1.0-win32.zip -OutFile glew-2.1.0-win32.zip
rem 
rem cd "%BUILD_DIR%/glew/extracted"
rem cmake -E tar xfvj ../glew-2.1.0-win32.zip
rem cd ..
rem 
rem cmake -E copy_directory "%BUILD_DIR%/glew/extracted/glew-2.1.0/include"         "%INSTALL_DIR%/include" || exit /b
rem cmake -E copy_directory "%BUILD_DIR%/glew/extracted/glew-2.1.0/lib/Release/x64" "%INSTALL_DIR%/lib"     || exit /b
rem cmake -E copy_directory "%BUILD_DIR%/glew/extracted/glew-2.1.0/bin/Release/x64" "%INSTALL_DIR%/bin"     || exit /b
rem 
rem rem  freeglut ---------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Building and installing freeglut ...
rem echo.
rem 
rem cmake -E make_directory "%BUILD_DIR%/freeglut" && cd "%BUILD_DIR%/freeglut"
rem cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem       -DCMAKE_INSTALL_LIBDIR=lib^
rem       "%EXTERNALS_DIR%/freeglut/freeglut/freeglut" || exit /b
rem 
rem cmake --build . --config %BUILD_TYPE% --target install --parallel 8
rem 
rem cmake -E copy_directory "%EXTERNALS_DIR%/freeglut/freeglut/freeglut/include/GL" "%INSTALL_DIR%/include/GL"
rem 
rem rem c-ares -----------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Building and installing c-ares ...
rem echo.
rem 
rem cmake -E make_directory "%BUILD_DIR%/c-ares" && cd "%BUILD_DIR%/c-ares"
rem cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem       "%EXTERNALS_DIR%/c-ares" || exit /b
rem 
rem cmake --build . --config %BUILD_TYPE% --target install --parallel 8
rem 
rem rem curl -------------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Building and installing curl ...
rem echo.
rem 
rem cmake -E make_directory "%BUILD_DIR%/curl" && cd "%BUILD_DIR%/curl"
rem cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem       -DBUILD_TESTING=OFF -DBUILD_CURL_EXE=OFF -DENABLE_ARES=ON^
rem       -DCARES_INCLUDE_DIR="%INSTALL_DIR%/include"^
rem       -DCARES_LIBRARY="%INSTALL_DIR%/lib/cares.lib"^
rem       -DCMAKE_INSTALL_LIBDIR=lib^
rem       "%EXTERNALS_DIR%/curl" || exit /b
rem 
rem cmake --build . --config %BUILD_TYPE% --target install --parallel 8
rem 
rem rem curlpp -----------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Building and installing curlpp ...
rem echo.
rem 
rem if "%COSMOSCOUT_DEBUG_BUILD%"=="true" (
rem   set CURL_LIB=libcurl-d_imp.lib
rem ) else ( 
rem   set CURL_LIB=libcurl_imp.lib
rem )
rem cmake -E make_directory "%BUILD_DIR%/curlpp" && cd "%BUILD_DIR%/curlpp"
rem cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem       -DCURL_INCLUDE_DIR="%INSTALL_DIR%/include"^
rem       -DCURL_LIBRARY="%INSTALL_DIR%/lib/%CURL_LIB%"^
rem       -DCMAKE_INSTALL_LIBDIR=lib^
rem       "%EXTERNALS_DIR%/curlpp" || exit /b
rem 
rem cmake --build . --config %BUILD_TYPE% --target install --parallel 8
rem 
rem rem libtiff ----------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Building and installing libtiff ...
rem echo.
rem 
rem cmake -E make_directory "%BUILD_DIR%/libtiff" && cd "%BUILD_DIR%/libtiff"
rem cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem       -DCMAKE_INSTALL_FULL_LIBDIR=lib^
rem       "%EXTERNALS_DIR%/libtiff" || exit /b
rem 
rem cmake --build . --config %BUILD_TYPE% --target install --parallel 8
rem 
rem rem gli --------------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Installing gli ...
rem echo.
rem 
rem cmake -E copy_directory "%EXTERNALS_DIR%/gli/gli" "%INSTALL_DIR%/include/gli" || exit /b
rem 
rem rem glm --------------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Installing glm ...
rem echo.
rem 
rem cmake -E copy_directory "%EXTERNALS_DIR%/glm/glm" "%INSTALL_DIR%/include/glm" || exit /b
rem 
rem rem tinygltf ---------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Installing tinygltf ...
rem echo.
rem 
rem cmake -E copy "%EXTERNALS_DIR%/tinygltf/json.hpp"          "%INSTALL_DIR%/include" || exit /b
rem cmake -E copy "%EXTERNALS_DIR%/tinygltf/stb_image.h"       "%INSTALL_DIR%/include" || exit /b
rem cmake -E copy "%EXTERNALS_DIR%/tinygltf/stb_image_write.h" "%INSTALL_DIR%/include" || exit /b
rem cmake -E copy "%EXTERNALS_DIR%/tinygltf/tiny_gltf.h"       "%INSTALL_DIR%/include" || exit /b
rem 
rem rem opensg -----------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Building and installing opensg-1.8 ...
rem echo.
rem 
rem cmake -E make_directory "%BUILD_DIR%/opensg-1.8" && cd "%BUILD_DIR%/opensg-1.8"
rem cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem       -DGLUT_INCLUDE_DIR="%INSTALL_DIR%/include" -DGLUT_LIBRARY="%INSTALL_DIR%/lib/freeglut.lib"^
rem       -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DOPENSG_BUILD_TESTS=Off "%EXTERNALS_DIR%/opensg-1.8"
rem 
rem cmake --build . --config %BUILD_TYPE% --target install --parallel 8
rem 
rem rem vista ------------------------------------------------------------------------------------------
rem 
rem echo.
rem echo Building and installing vista ...
rem echo.
rem 
rem cmake -E make_directory "%BUILD_DIR%/vista" && cd "%BUILD_DIR%/vista"
rem 
rem rem set OPENVR="T:/modulesystem/tools/openvr/OpenVR_SDK_1.0.3/install/win7.x86_64.msvc14.release"
rem rem cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem rem       -DVISTACORELIBS_USE_VIVE=On -DVISTADRIVERS_BUILD_VIVE=On -DOPENVR_ROOT_DIR=%OPENVR%^
rem rem       -DVISTADRIVERS_BUILD_3DCSPACENAVIGATOR=On^
rem rem       -DCMAKE_CXX_FLAGS="-std=c++11" "%EXTERNALS_DIR%/vista" || exit /b
rem 
rem cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem       -DCMAKE_CXX_FLAGS="-std=c++11" "%EXTERNALS_DIR%/vista" || exit /b
rem 
rem cmake --build . --config %BUILD_TYPE% --target install --parallel 8
rem 
rem rem cspice -----------------------------------------------------------------------------------------
rem 
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

cmake %CMAKE_FLAGS% . || exit /b

cmake --build . --config %BUILD_TYPE% --parallel 8 || exit /b

cmake -E copy_directory "%BUILD_DIR%/cspice/extracted/cspice/include"            "%INSTALL_DIR%/include/cspice"
cmake -E copy           "%BUILD_DIR%/cspice/extracted/cspice/%BUILD_TYPE%/cspice.lib" "%INSTALL_DIR%/lib"
cmake -E copy           "%BUILD_DIR%/cspice/extracted/cspice/%BUILD_TYPE%/cspice.dll" "%INSTALL_DIR%/lib"

rem cef --------------------------------------------------------------------------------------------

echo.
echo Downloading bzip2 ...
echo.

cmake -E make_directory "%BUILD_DIR%/cef/bzip2" && cd "%BUILD_DIR%/cef"
powershell.exe -command Invoke-WebRequest -Uri https://netcologne.dl.sourceforge.net/project/gnuwin32/bzip2/1.0.5/bzip2-1.0.5-bin.zip -OutFile bzip2.zip
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

rem We don't want the example applications.
rmdir %CEF_VERSION%\tests /s /q

rem We want to built with /MD
powershell.exe -command "(gc %CEF_VERSION%\cmake\cef_variables.cmake) -replace '/MT', '/MD' | Out-File -encoding UTF8 %CEF_VERSION%\cmake\cef_variables.cmake"

cd ..

cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      "%BUILD_DIR%/cef/extracted/%CEF_VERSION%" || exit /b

cmake --build . --config %BUILD_TYPE% --parallel 8 || exit /b     

echo Installing cef...
cmake -E make_directory "%INSTALL_DIR%/include/cef"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_VERSION%/include"               "%INSTALL_DIR%/include/cef/include"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_VERSION%/Resources"             "%INSTALL_DIR%/share/cef"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_VERSION%/Release"               "%INSTALL_DIR%/lib"
cmake -E copy "%BUILD_DIR%/cef/libcef_dll_wrapper/%BUILD_TYPE%/libcef_dll_wrapper.lib"  "%INSTALL_DIR%/lib"


rem ------------------------------------------------------------------------------------------------

cd "%CURRENT_DIR%"
echo Finished successfully.

@echo on
