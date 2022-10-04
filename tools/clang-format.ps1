# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

<#
    .SYNOPSIS
    This function runs clang-format on all C++ and JavaScript source files.

    .DESCRIPTION
    This function runs clang-format on all C++ and JavaScript source files.
#>

$sourceDir = "$PSScriptRoot/../src"
$pluginDir = "$PSScriptRoot/../plugins"

$fileEndings = @('*.cpp', '*.hpp', '*.inl', '*.js')

$itemsToCheck = Get-ChildItem -Path $sourceDir, $pluginDir -Recurse -Include $fileEndings `
                              | Where-Object { $_.FullName -notmatch "third-party" }

try {
    # If we have a recent PowerShell version we can run clang-format in parallel
    # which is much faster. But we still need to support PowerShell version 5.
    $parallelSupported = $PSVersionTable.PSVersion.Major -ge 7
    if ($parallelSupported) {
        $itemsToCheck | ForEach-Object -Parallel {
            $file = $_
            Write-Output "Formatting $file ..."
            clang-format -i "$file"
        }
    } else {
        $itemsToCheck | ForEach-Object {
            $file = $_
            Write-Output "Formatting $file ..."
            clang-format -i "$file"
        }
    }
} catch {
    throw $_
}