# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

<#
   .SYNOPSIS
   This function runs clang-tidy on all source files.

   .DESCRIPTION
   For this script to work CosmoScout has to build at least once with a compile_commands.json file
   in the build directory. You can provide a build directory or the script searches common paths
   for one.
   Not that on Windows you have to use the Ninja generator to get a compile_commands.json file.
   Look here for more information regarding the compile_commands.json file:
   https://cmake.org/cmake/help/latest/variable/CMAKE_EXPORT_COMPILE_COMMANDS.html

   .PARAMETER BuildDirectory
   The path to the build of CosmoScout VR. The directory must contain a compile_commands.json file.
   If this parameter is not set by the user the build and ../build folders are being searched for
   a directory with a compile_commands.json in it.

   .PARAMETER Quiet
   Enables the -quiet flag in clang-tidy.

   .PARAMETER Fix
   Enables the -fix flag in clang-tidy which automatically fixes found issues where it can.
#>

param(
    [String]
    $BuildDirectory,

    [Boolean]
    $Quiet = $true,

    [Boolean]
    $Fix = $true
)

if (-not $BuildDirectory) {
    # PowerShell only supports OsChecks from version 6 onwards. Linux PowerShell scripts are always
    # version 6+, so the check below always assigns the correct name.
    $osName = "windows"
    if ($IsLinux) {
        $osName = "linux"
    }

    $root        = "$PSScriptRoot/.."
    $releasePath = "build/$osName-Release"
    $debugPath   = "build/$osName-Debug"
    $cc          = "compile_commands.json"

    # Search the common directories first.
    $BuildDirectory = if (Test-Path "$root/$releasePath/$cc") {
        "$root/$releasePath"
    } elseif (Test-Path "$root/$debugPath/$cc") {
        "$root/$debugPath"
    } elseif (Test-Path "$root/../$releasePath/$cc") {
        "$root/../$releasePath"
    } elseif (Test-Path "$root/../$debugPath/$cc") {
        "$root/../$debugPath"
    } else {
        # The common directories didn't provide any results, so we try to search for any
        # compile_commands.json. This takes really long with lots of build directories.
        $directories = Get-ChildItem -Path "$root/build", "$root/../build" -Recurse -Filter "$cc"

        if ($directories.Length -gt 0) {
            $directories[0].Directory.FullName
        }
    }
}

function Invoke-Clang-Tidy($path) {
    # Python doesn't search the system path for scripts, so we get the absolute path from the OS.
    $clangTidyPath = $(Get-Command run-clang-tidy.py).Path

    if (Test-Path $clangTidyPath) {
        python $clangTidyPath $(if ($Quiet) { "-quiet" }) $(if ($Fix) { "-fix" }) -p "$path"
    } else {
        Write-Warning "Failed to find run-clang-tidy.py!
                       Ensure that run-clang-tidy.py is on the system path."
    }
}

if (Test-Path $BuildDirectory) {
    Invoke-Clang-Tidy $BuildDirectory
} else {
    Write-Warning "Failed to find build directory!
                   The directory must contain a compile_commands.json file."
}