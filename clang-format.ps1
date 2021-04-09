# ------------------------------------------------------------------------------------------------ #
#                                 This file is part of CosmoScout VR                               #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

<#
    .SYNOPSIS
    This function runs clang-format on all C++ and JavaScript source files.

    .DESCRIPTION
    This function runs clang-format on all C++ and JavaScript source files.
#>

$sourceDir = "$PSScriptRoot/src"
$pluginDir = "$PSScriptRoot/plugins"

$fileEndings = @('*.cpp', '*.hpp', '*.inl', '*.js')

$itemsToCheck = Get-ChildItem -Path $sourceDir, $pluginDir -Recurse -Include $fileEndings `
                              | Where-Object { $_.FullName -notmatch "third-party" }

try {
    # If we have a recent PowerShell version we can run clang-format in parallel
    # which is much faster. But we still need to support PowerShell version 5.
    if ($PSVersionTable.PSVersion.Major -ge 7) {
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