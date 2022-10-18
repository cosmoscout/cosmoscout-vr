# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

<#
    .SYNOPSIS
    This function creates a new CosmoScout VR plugin library with the given name.
    
    .DESCRIPTION
    This function creates a new CosmoScout VR plugin library with the given name. It will generate
    all relevant files for
    your plugin library:
    - README.md
    - CMakeLists.txt
    - src/logger.hpp
    - src/logger.cpp
    
    .PARAMETER Name
    The plugin library name. Just write it out in natural language. Don't use CamelCase, KebabCase, 
    PascalCase or SnakeCase. The script will convert it to the correct case for each occurance.

    .EXAMPLE
    PS> .\New-PluginLibrary -Name "Hello World"

    .EXAMPLE
    PS> .\New-PluginLibrary "Hello World"
#>

param (
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [String]
    $Name
)

$lowerCaseName = $Name.ToLower()
$upperCaseName = $Name.ToUpper()
$titleCaseName = (Get-Culture).TextInfo.ToTitleCase($Name)

$kebabCaseName = $lowerCaseName -replace ' ','-'
$lowerCaseJoinedName = $lowerCaseName -replace ' '
$pascalCaseName = $titleCaseName -replace ' '
$camelCaseName = $pascalCaseName.Substring(0, 1).ToLower() + $pascalCaseName.Substring(1)
$screamingSnakeCaseName = $upperCaseName -replace ' ','_'
$snakeCaseName = $lowerCaseName -replace ' ','_'

$pluginRootDir = "plugins/csl-$kebabCaseName"

function New-PluginLibrary {
    $pluginDir = New-Item -Path "$PSScriptRoot/../plugins" -Name "csl-$kebabCaseName" -ItemType "directory"

    New-README $pluginDir
    New-CMakeLists $pluginDir

    $srcDir = New-Item -Path $pluginDir -Name "src" -ItemType "directory"
    New-CppLogger $srcDir
}

function New-README($directory) {
    $readMeText = @"
# $titleCaseName for CosmoScout VR

Put the plugin library description here!
``````
"@
    Write-Host "Creating: '$pluginRootDir/README.md'"
    $readMeText | Out-File "$directory/README.md" -Encoding utf8
}

function New-CMakeLists($directory) {
    $cMakeListsText = @"
# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

option(CSL_$screamingSnakeCaseName `"Enable compilation of this plugin library`" ON)

if (NOT CSL_$screamingSnakeCaseName)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)

add_library(csl-$kebabCaseName SHARED
  `${SOURCE_FILES}
  `${HEADER_FILES}
)

target_link_libraries(csl-$kebabCaseName
  PUBLIC
    cs-core
)

# Add this plugin library to a `"plugins`" folder in your IDE.
set_property(TARGET csl-$kebabCaseName PROPERTY FOLDER `"plugins`")

# Make directory structure available in your IDE.
source_group(TREE `"`${CMAKE_CURRENT_SOURCE_DIR}`" FILES
  `${SOURCE_FILES} `${HEADER_FILES}
)

target_include_directories(csl-$kebabCaseName PUBLIC
    ${CMAKE_BINARY_DIR}/plugins/csl-$kebabCaseName
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS csl-$kebabCaseName DESTINATION `"lib`")

# export header ------------------------------------------------------------------------------------

generate_export_header(csl-$kebabCaseName
    EXPORT_FILE_NAME csl_${snakeCaseName}_export.hpp
)
"@

    Write-Host "Creating: '$pluginRootDir/CMakeLists.txt'"
    $cMakeListsText | Out-File "$directory/CMakeLists.txt" -Encoding utf8
}

function New-CppLogger($directory) {
    $loggerHeaderText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_${screamingSnakeCaseName}_LOGGER_HPP
#define CSL_${screamingSnakeCaseName}_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csl::$lowerCaseJoinedName {

/// This creates the default singleton logger for "csl-$kebabCaseName" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csl::$lowerCaseJoinedName

#endif // CSL_${screamingSnakeCaseName}_LOGGER_HPP
"@

    $loggerSourceText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "logger.hpp"

#include "../../../src/cs-utils/logger.hpp"

namespace csl::$lowerCaseJoinedName {

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::logger& logger() {
  static auto logger = cs::utils::createLogger("csl-$kebabCaseName");
  return *logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csl::$lowerCaseJoinedName
"@

    Write-Host "Creating: '$pluginRootDir/src/logger.hpp'"
    $loggerHeaderText | Out-File "$directory/logger.hpp" -Encoding utf8

    Write-Host "Creating: '$pluginRootDir/src/logger.cpp'"
    $loggerSourceText | Out-File "$directory/logger.cpp" -Encoding utf8
}

New-PluginLibrary
