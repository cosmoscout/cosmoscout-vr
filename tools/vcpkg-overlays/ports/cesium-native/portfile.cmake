# SPDX-FileCopyrightText: Timothy Moore
# SPDX-License-Identifier: MIT

if(VCPKG_TARGET_IS_WINDOWS)
  vcpkg_check_linkage(ONLY_STATIC_LIBRARY)
endif()

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
    dependencies-only CESIUM_NATIVE_DEPS_ONLY
)

if(CESIUM_NATIVE_DEPS_ONLY)
  message(STATUS "skipping installation of cesium-native")
  set(VCPKG_POLICY_EMPTY_PACKAGE enabled)
  return()
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO CesiumGS/cesium-native
    REF "v${VERSION}"
    SHA512 2b2595107e87a31452773d4b52244a86fd0ab2107dcdcf2fabe4487bf00c9973413e0dbe34e2477c798d444150e948d9eda145e9cee5d95e02523d84b60daceb
    HEAD_REF main
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
    -DCESIUM_USE_EZVCPKG=OFF
    -DCESIUM_TESTS_ENABLED=OFF
    -DCESIUM_ENABLE_CLANG_TIDY=OFF
    --compile-no-warning-as-error
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH share/cesium-native/cmake PACKAGE_NAME cesium-native)
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")