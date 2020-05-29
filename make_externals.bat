@echo off

rem ---------------------------------------------------------------------------------------------- #
rem                              This file is part of CosmoScout VR                                #
rem     and may be used under the terms of the MIT license. See the LICENSE file for details.      #
rem                       Copyright: (c) 2019 German Aerospace Center (DLR)                        #
rem ---------------------------------------------------------------------------------------------- #

rem ---------------------------------------------------------------------------------------------- #
rem Make sure to run "git submodule update --init" before executing this script!                   #
rem Default build mode is release, if "set COSMOSCOUT_DEBUG_BUILD=true" is executed before, all    #
rem dependencies will be built in debug mode.                                                      #
rem Usage:                                                                                         #
rem    make_externals.bat [additional CMake flags, defaults to -G "Visual Studio 15 Win64"]        #
rem Examples:                                                                                      #
rem    make_externals.bat                                                                          #
rem    make_externals.bat -G "Visual Studio 15 Win64"                                              #
rem    make_externals.bat -G "Visual Studio 16 2019" -A x64                                        #
rem    make_externals.bat -GNinja -DCMAKE_C_COMPILER=cl.exe -DCMAKE_CXX_COMPILER=cl.exe            #
rem ---------------------------------------------------------------------------------------------- #

rem The CMake generator and other flags can be passed as parameters.
set CMAKE_FLAGS=-G "Visual Studio 15 Win64"
IF NOT "%~1"=="" (
  SET CMAKE_FLAGS=%*
)

rem We need to check if Ninja is used as a generator, since there are some minor differences in generation paths.
echo.%CMAKE_FLAGS%|findstr /C:"Ninja" >nul 2>&1
IF NOT errorlevel 1 (
   set USING_NINJA=true
) else (
   set USING_NINJA=false
)

rem Check if ComoScout VR debug build is enabled with "set COSMOSCOUT_DEBUG_BUILD=true".
IF "%COSMOSCOUT_DEBUG_BUILD%"=="true" (
  echo CosmoScout VR debug build is enabled!
  set BUILD_TYPE=Debug
) else (
  set BUILD_TYPE=Release
)

rem Check if unity build is disabled with "set COSMOSCOUT_NO_UNITY_BUILD=true".
IF "%COSMOSCOUT_NO_UNITY_BUILD%"=="true" (
  echo Unity build is disabled!
  set UNITY_BUILD=Off
) else (
  set UNITY_BUILD=On
)

rem Check if precompield headers should not be used with "set COSMOSCOUT_NO_PCH=true".
IF "%COSMOSCOUT_NO_PCH%"=="true" (
  echo Precompiled headers are disabled!
  set PRECOMPILED_HEADERS=Off
) else (
  set PRECOMPILED_HEADERS=On
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

rem glew -------------------------------------------------------------------------------------------
:glew

echo.
echo Downloading, building and installing GLEW ...
echo.

cmake -E make_directory "%BUILD_DIR%/glew/extracted" && cd "%BUILD_DIR%/glew"

IF NOT EXIST glew-2.1.0-win32.zip (
  powershell.exe -command Invoke-WebRequest -Uri https://netcologne.dl.sourceforge.net/project/glew/glew/2.1.0/glew-2.1.0-win32.zip -OutFile glew-2.1.0-win32.zip
) else (
  echo File 'glew-2.1.0-win32.zip' already exists, no download required.
)

cd "%BUILD_DIR%/glew/extracted"
cmake -E tar xfvj ../glew-2.1.0-win32.zip
cd ..

cmake -E copy_directory "%BUILD_DIR%/glew/extracted/glew-2.1.0/include"         "%INSTALL_DIR%/include" || exit /b
cmake -E copy_directory "%BUILD_DIR%/glew/extracted/glew-2.1.0/lib/Release/x64" "%INSTALL_DIR%/lib"     || exit /b
cmake -E copy_directory "%BUILD_DIR%/glew/extracted/glew-2.1.0/bin/Release/x64" "%INSTALL_DIR%/bin"     || exit /b

rem  freeglut ---------------------------------------------------------------------------------------
:freeglut

echo.
echo Building and installing freeglut ...
echo.

cmake -E make_directory "%BUILD_DIR%/freeglut" && cd "%BUILD_DIR%/freeglut"
cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DFREEGLUT_BUILD_DEMOS=Off -DCMAKE_INSTALL_LIBDIR=lib -DFREEGLUT_BUILD_STATIC_LIBS=Off^
      "%EXTERNALS_DIR%/freeglut/freeglut/freeglut" || exit /b

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS%

cmake -E copy_directory "%EXTERNALS_DIR%/freeglut/freeglut/freeglut/include/GL" "%INSTALL_DIR%/include/GL"

rem c-ares -----------------------------------------------------------------------------------------
:c-ares

echo.
echo Building and installing c-ares ...
echo.

cmake -E make_directory "%BUILD_DIR%/c-ares" && cd "%BUILD_DIR%/c-ares"
cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCARES_BUILD_TOOLS=OFF^
      -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      "%EXTERNALS_DIR%/c-ares" || exit /b

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS%

rem curl -------------------------------------------------------------------------------------------
:curl

echo.
echo Building and installing curl ...
echo.

cmake -E make_directory "%BUILD_DIR%/curl" && cd "%BUILD_DIR%/curl"
cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_BUILD_TYPE=%BUILD_TYPE%^
      -DBUILD_TESTING=OFF -DBUILD_CURL_EXE=OFF -DENABLE_ARES=ON^
      -DCARES_INCLUDE_DIR="%INSTALL_DIR%/include"^
      -DCARES_LIBRARY="%INSTALL_DIR%/lib/cares.lib"^
      -DCMAKE_USE_WINSSL=On -DCMAKE_INSTALL_LIBDIR=lib^
      "%EXTERNALS_DIR%/curl" || exit /b

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS%

rem curlpp -----------------------------------------------------------------------------------------
:curlpp

echo.
echo Building and installing curlpp ...
echo.

if "%COSMOSCOUT_DEBUG_BUILD%"=="true" (
  set CURL_LIB=libcurl-d_imp.lib
) else ( 
  set CURL_LIB=libcurl_imp.lib
)
cmake -E make_directory "%BUILD_DIR%/curlpp" && cd "%BUILD_DIR%/curlpp"
cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" -DCMAKE_UNITY_BUILD=%UNITY_BUILD%^
      -DCMAKE_BUILD_TYPE=%BUILD_TYPE%^
      -DCURL_INCLUDE_DIR="%INSTALL_DIR%/include"^
      -DCURL_LIBRARY="%INSTALL_DIR%/lib/%CURL_LIB%"^
      -DCMAKE_INSTALL_LIBDIR=lib -DCURL_NO_CURL_CMAKE=On^
      "%EXTERNALS_DIR%/curlpp" || exit /b

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS%

rem libtiff ----------------------------------------------------------------------------------------
:libtiff

echo.
echo Building and installing libtiff ...
echo.

cmake -E make_directory "%BUILD_DIR%/libtiff" && cd "%BUILD_DIR%/libtiff"
cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" -DCMAKE_UNITY_BUILD=%UNITY_BUILD%^
      -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DBUILD_SHARED_LIBS=Off -DCMAKE_INSTALL_FULL_LIBDIR=lib^
      "%EXTERNALS_DIR%/libtiff" || exit /b

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS%

rem spdlog -----------------------------------------------------------------------------------------
:spdlog

echo.
echo Building and installing spdlog ...
echo.

cmake -E make_directory "%BUILD_DIR%/spdlog" && cd "%BUILD_DIR%/spdlog"
cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DSPDLOG_ENABLE_PCH=On "%EXTERNALS_DIR%/spdlog" || exit /b

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS%

rem civetweb -----------------------------------------------------------------------------------------
:civetweb

echo.
echo Building and installing civetweb ...
echo.

cmake -E make_directory "%BUILD_DIR%/civetweb" && cd "%BUILD_DIR%/civetweb"
cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCIVETWEB_BUILD_TESTING=OFF -DCIVETWEB_ENABLE_SERVER_EXECUTABLE=OFF^
      -DCIVETWEB_ENABLE_CXX=On -DBUILD_SHARED_LIBS=On "%EXTERNALS_DIR%/civetweb" || exit /b

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS%

rem jsonhpp ----------------------------------------------------------------------------------------
:jsonhpp

echo.
echo Installing jsonHPP ...
echo.

cmake -E copy_directory "%EXTERNALS_DIR%/json/include/nlohmann" "%INSTALL_DIR%/include/nlohmann" || exit /b

rem doctest ----------------------------------------------------------------------------------------
:doctest

echo.
echo Installing doctest ...
echo.

cmake -E copy_directory "%EXTERNALS_DIR%/doctest/doctest" "%INSTALL_DIR%/include/doctest" || exit /b

rem gli --------------------------------------------------------------------------------------------
:gli

echo.
echo Installing gli ...
echo.

cmake -E copy_directory "%EXTERNALS_DIR%/gli/gli" "%INSTALL_DIR%/include/gli" || exit /b

rem glm --------------------------------------------------------------------------------------------
:glm

echo.
echo Installing glm ...
echo.

cmake -E copy_directory "%EXTERNALS_DIR%/glm/glm" "%INSTALL_DIR%/include/glm" || exit /b

rem tinygltf ---------------------------------------------------------------------------------------
:tinygltf

echo.
echo Installing tinygltf ...
echo.

cmake -E copy "%EXTERNALS_DIR%/tinygltf/json.hpp"    "%INSTALL_DIR%/include" || exit /b
cmake -E copy "%EXTERNALS_DIR%/tinygltf/tiny_gltf.h" "%INSTALL_DIR%/include" || exit /b

rem stb --------------------------------------------------------------------------------------------
:stb

echo.
echo Installing stb ...
echo.

cmake -E copy "%EXTERNALS_DIR%/stb/stb_image.h"        "%INSTALL_DIR%/include" || exit /b
cmake -E copy "%EXTERNALS_DIR%/stb/stb_image_write.h"  "%INSTALL_DIR%/include" || exit /b
cmake -E copy "%EXTERNALS_DIR%/stb/stb_image_resize.h" "%INSTALL_DIR%/include" || exit /b

rem opensg -----------------------------------------------------------------------------------------
:opensg

echo.
echo Building and installing opensg-1.8 ...
echo.

cmake -E make_directory "%BUILD_DIR%/opensg-1.8" && cd "%BUILD_DIR%/opensg-1.8"
cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_UNITY_BUILD=%UNITY_BUILD%^
      -DOPENSG_USE_PRECOMPILED_HEADERS=%PRECOMPILED_HEADERS%^
      -DGLUT_INCLUDE_DIR="%INSTALL_DIR%/include" -DGLUT_LIBRARY="%INSTALL_DIR%/lib/freeglut.lib"^
      -DCMAKE_SHARED_LINKER_FLAGS="/FORCE:MULTIPLE" -DOPENSG_BUILD_TESTS=Off^
      "%EXTERNALS_DIR%/opensg-1.8"

cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS%

rem vista ------------------------------------------------------------------------------------------
:vista

echo.
echo Building and installing vista ...
echo.

cmake -E make_directory "%BUILD_DIR%/vista" && cd "%BUILD_DIR%/vista"

rem set OPENVR="T:/modulesystem/tools/openvr/OpenVR_SDK_1.0.3/install/win7.x86_64.msvc14.release"
rem cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
rem       -DVISTACORELIBS_USE_VIVE=On -DVISTADRIVERS_BUILD_VIVE=On -DOPENVR_ROOT_DIR=%OPENVR%^
rem       -DVISTADRIVERS_BUILD_3DCSPACENAVIGATOR=On^
rem       -DCMAKE_CXX_FLAGS="-std=c++11" "%EXTERNALS_DIR%/vista" || exit /b

cmake %CMAKE_FLAGS% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%" -DVISTADEMO_ENABLED=Off^
      -DCMAKE_BUILD_TYPE=%BUILD_TYPE%^
      -DCMAKE_UNITY_BUILD=%UNITY_BUILD% -DVISTA_USE_PRECOMPILED_HEADERS=%PRECOMPILED_HEADERS%^
      "%EXTERNALS_DIR%/vista" || exit /b
cmake --build . --config %BUILD_TYPE% --target install --parallel %NUMBER_OF_PROCESSORS%

rem cspice -----------------------------------------------------------------------------------------
:cspice

echo.
echo Downloading and installing cspice ...
echo.

cmake -E make_directory "%BUILD_DIR%/cspice/extracted" && cd "%BUILD_DIR%/cspice"

IF NOT EXIST cspice.zip (
  powershell.exe -command $AllProtocols = [System.Net.SecurityProtocolType]'Tls11,Tls12'; [System.Net.ServicePointManager]::SecurityProtocol = $AllProtocols; Invoke-WebRequest -Uri https://naif.jpl.nasa.gov/pub/naif/toolkit//C/PC_Windows_VisualC_64bit/packages/cspice.zip -OutFile cspice.zip
) else (
  echo File 'cspice.zip' already exists no, download required.
)

cd "%BUILD_DIR%/cspice/extracted"
cmake -E tar xfvj ../cspice.zip
cd cspice

echo project(cspice C) > "CMakeLists.txt"
echo cmake_minimum_required(VERSION 2.8) >> "CMakeLists.txt"
echo add_definitions("-D_COMPLEX_DEFINED -DMSDOS -DOMIT_BLANK_CC -DKR_headers -DNON_ANSI_STDIO") >> "CMakeLists.txt"
echo file(GLOB_RECURSE CSPICE_SOURCE src/cspice/*.c) >> "CMakeLists.txt"
echo add_library(cspice SHARED ${CSPICE_SOURCE}) >> "CMakeLists.txt"
echo set_target_properties(cspice PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS 1) >> "CMakeLists.txt"

cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% . || exit /b

cmake --build . --config %BUILD_TYPE% --parallel %NUMBER_OF_PROCESSORS% || exit /b

cmake -E copy_directory "%BUILD_DIR%/cspice/extracted/cspice/include" "%INSTALL_DIR%/include/cspice"

if %USING_NINJA%==true (
  cmake -E copy "%BUILD_DIR%/cspice/extracted/cspice/cspice.lib" "%INSTALL_DIR%/lib"
  cmake -E copy "%BUILD_DIR%/cspice/extracted/cspice/cspice.dll" "%INSTALL_DIR%/lib"
) else (
  cmake -E copy "%BUILD_DIR%/cspice/extracted/cspice/%BUILD_TYPE%/cspice.lib" "%INSTALL_DIR%/lib"
  cmake -E copy "%BUILD_DIR%/cspice/extracted/cspice/%BUILD_TYPE%/cspice.dll" "%INSTALL_DIR%/lib"
)

rem cef --------------------------------------------------------------------------------------------
:cef

echo.
echo Downloading bzip2 ...
echo.

cmake -E make_directory "%BUILD_DIR%/cef/bzip2" && cd "%BUILD_DIR%/cef"

IF NOT EXIST bzip2.zip (
  powershell.exe -command Invoke-WebRequest -Uri https://netcologne.dl.sourceforge.net/project/gnuwin32/bzip2/1.0.5/bzip2-1.0.5-bin.zip -OutFile bzip2.zip
) else (
  echo File 'bzip2.zip' already exists, no download required.
)

cd "%BUILD_DIR%/cef/bzip2"
cmake -E tar xfvj ../bzip2.zip
cd ..

echo.
echo Downloading, building and installing cef (this may take some time) ...
echo.

set CEF_DIR=cef_binary_81.3.3+g072a5f5+chromium-81.0.4044.138_windows64_minimal

cmake -E make_directory "%BUILD_DIR%/cef/extracted" && cd "%BUILD_DIR%/cef"

IF NOT EXIST cef.tar (
  powershell.exe -command Invoke-WebRequest -Uri http://opensource.spotify.com/cefbuilds/cef_binary_81.3.3%%2Bg072a5f5%%2Bchromium-81.0.4044.138_windows64_minimal.tar.bz2 -OutFile cef.tar.bz2
  cd "%BUILD_DIR%/cef/extracted"
  "%BUILD_DIR%/cef/bzip2/bin/bunzip2.exe" -v ../cef.tar.bz2
) else (
  echo File 'cef.tar' already exists, no download required.
)

cd "%BUILD_DIR%/cef/extracted"
cmake -E tar xfvj ../cef.tar

rem We don't want the example applications.
rmdir %CEF_DIR%\tests /s /q

cd ..

cmake %CMAKE_FLAGS% -DCMAKE_BUILD_TYPE=%BUILD_TYPE% -DCMAKE_INSTALL_PREFIX="%INSTALL_DIR%"^
      -DCMAKE_UNITY_BUILD=%UNITY_BUILD% -DCEF_RUNTIME_LIBRARY_FLAG=/MD -DCEF_DEBUG_INFO_FLAG=""^
      "%BUILD_DIR%/cef/extracted/%CEF_DIR%" || exit /b

cmake --build . --config %BUILD_TYPE% --parallel %NUMBER_OF_PROCESSORS% || exit /b

echo Installing cef...
cmake -E make_directory "%INSTALL_DIR%/include/cef"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_DIR%/include"                   "%INSTALL_DIR%/include/cef/include"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_DIR%/Resources"                 "%INSTALL_DIR%/share/cef"
cmake -E copy_directory "%BUILD_DIR%/cef/extracted/%CEF_DIR%/Release"                   "%INSTALL_DIR%/lib"

if %USING_NINJA%==true (
  cmake -E copy "%BUILD_DIR%/cef/libcef_dll_wrapper/libcef_dll_wrapper.lib"  "%INSTALL_DIR%/lib"
) else (
  cmake -E copy "%BUILD_DIR%/cef/libcef_dll_wrapper/%BUILD_TYPE%/libcef_dll_wrapper.lib"  "%INSTALL_DIR%/lib"
)

rem ------------------------------------------------------------------------------------------------

:finish

cd "%CURRENT_DIR%"
echo Finished successfully.

@echo on
