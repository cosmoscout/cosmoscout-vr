# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO cosmoscout/opensg-1.8
    REF 2afc1bd51a82052efb6566e21409d40e3ab81427
    SHA512 a79fb2d898b49208cd436772aa1805e3b83912acb7b0e273a39e32b7871f238ae137bdae0f0b1a07db60f804ddab90e20f7af72c49c2fe67247955673c7e974c
    HEAD_REF feature/vcpkg
)

set(OPENSG_WINDOW_FEATURE Off)
if ("window" IN_LIST FEATURES)
    set(OPENSG_WINDOW_FEATURE On)
endif ()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DOPENSG_BUILD_TESTS=Off
        -DOPENSG_BUILD_WINDOW=${OPENSG_WINDOW_FEATURE}
        -DOPENSG_USE_PRECOMPILED_HEADERS=On
        -DOPENSG_INFINITE_REVERSE_PROJECTION=On
        -DCMAKE_UNITY_BUILD=On
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
