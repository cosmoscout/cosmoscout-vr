# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

# 1. The Toggle: Allows developers to turn off compilation of this specific plugin
option(CSP_CESIUM_RENDERER "Enable compilation of this plugin" ON) 

if(NOT CSP_CESIUM_RENDERER)
    return()
    
endif()

# build plugin -------------------------------------------------------------------------------------

# 2. Define Files used in this Plugin Here (Did not use GLOB becasue it created ghost files which compiler missed)
set(SOURCE_FILES
  src/Plugin.cpp
  src/logger.cpp
  src/CesiumUtils.cpp
)

set(HEADER_FILES
  src/Plugin.hpp
  src/logger.hpp
  src/CesiumUtils.hpp
)

# 3. Create Target
add_library(csp-cesium-renderer SHARED
  ${SOURCE_FILES}
  ${HEADER_FILES}
)


# 4. Link CosmoScout Core Libraries
target_link_libraries(csp-cesium-renderer
  PUBLIC
    cs-core
)


# Add this Plugin to a "plugins" folder in your IDE.
set_property(TARGET csp-cesium-renderer PROPERTY FOLDER "plugins")

# Make directory structure available in your IDE.
source_group(TREE "${CMAKE_CURRENT_SOURCE_DIR}" FILES
  ${SOURCE_FILES} ${HEADER_FILES}
)

# install plugin -----------------------------------------------------------------------------------
install(TARGETS csp-cesium-renderer DESTINATION "share/plugins")


# Cesium Native paths
set(CESIUM_NATIVE_DIR "C:/Dev/CesiumSDK/cesium-native/dist")
set(CESIUM_VCPKG_DIR  "C:/Dev/CesiumSDK/cesium-native/build/vcpkg_installed/x64-windows-static-md-1434")

# Cesium include paths (SYSTEM = suppress warnings from their headers)
target_include_directories(csp-cesium-renderer SYSTEM PRIVATE
  ${CESIUM_NATIVE_DIR}/include
  ${CESIUM_VCPKG_DIR}/include
)

# Cesium Native libraries (dependency order: highest → lowest)
target_link_libraries(csp-cesium-renderer PRIVATE
  # -- Top-level: what YOUR code directly calls --
  ${CESIUM_NATIVE_DIR}/lib/Cesium3DTilesSelection.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumCurl.lib

  # -- Mid-level: pulled in by the above --
  ${CESIUM_NATIVE_DIR}/lib/Cesium3DTilesContent.lib
  ${CESIUM_NATIVE_DIR}/lib/Cesium3DTilesReader.lib
  ${CESIUM_NATIVE_DIR}/lib/Cesium3DTilesWriter.lib
  ${CESIUM_NATIVE_DIR}/lib/Cesium3DTiles.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumRasterOverlays.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumQuantizedMeshTerrain.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumVectorData.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumClientCommon.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumIonClient.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumITwinClient.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumGltfContent.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumGltfReader.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumGltfWriter.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumGltf.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumJsonReader.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumJsonWriter.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumGeospatial.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumGeometry.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumAsync.lib
  ${CESIUM_NATIVE_DIR}/lib/CesiumUtility.lib
)


# vcpkg transitive dependencies (required by Cesium's static libs)
target_link_libraries(csp-cesium-renderer PRIVATE
  ${CESIUM_VCPKG_DIR}/lib/async++.lib
  ${CESIUM_VCPKG_DIR}/lib/draco.lib
  ${CESIUM_VCPKG_DIR}/lib/s2.lib
  ${CESIUM_VCPKG_DIR}/lib/libcurl.lib
  ${CESIUM_VCPKG_DIR}/lib/libssl.lib
  ${CESIUM_VCPKG_DIR}/lib/libcrypto.lib
  ${CESIUM_VCPKG_DIR}/lib/sqlite3.lib
  ${CESIUM_VCPKG_DIR}/lib/tinyxml2.lib
  ${CESIUM_VCPKG_DIR}/lib/ada.lib
  ${CESIUM_VCPKG_DIR}/lib/meshoptimizer.lib
  ${CESIUM_VCPKG_DIR}/lib/libmodpbase64.lib
  ${CESIUM_VCPKG_DIR}/lib/ktx.lib
  ${CESIUM_VCPKG_DIR}/lib/spz.lib
  ${CESIUM_VCPKG_DIR}/lib/libwebp.lib
  ${CESIUM_VCPKG_DIR}/lib/libwebpdecoder.lib
  ${CESIUM_VCPKG_DIR}/lib/libsharpyuv.lib
  ${CESIUM_VCPKG_DIR}/lib/blend2d.lib
  ${CESIUM_VCPKG_DIR}/lib/asmjit.lib
  ${CESIUM_VCPKG_DIR}/lib/fmt.lib
  ${CESIUM_VCPKG_DIR}/lib/spdlog.lib
  ${CESIUM_VCPKG_DIR}/lib/zlib.lib
  ${CESIUM_VCPKG_DIR}/lib/zlibstatic-ng.lib
  ${CESIUM_VCPKG_DIR}/lib/zstd.lib
  ${CESIUM_VCPKG_DIR}/lib/brotlidec.lib
  ${CESIUM_VCPKG_DIR}/lib/brotlicommon.lib
  ${CESIUM_VCPKG_DIR}/lib/brotlienc.lib
  ${CESIUM_VCPKG_DIR}/lib/jpeg.lib
  ${CESIUM_VCPKG_DIR}/lib/turbojpeg.lib
  ${CESIUM_VCPKG_DIR}/lib/astcenc-avx2-static.lib
    # Abseil libraries (required by s2.lib)
  ${CESIUM_VCPKG_DIR}/lib/absl_base.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_spinlock_wait.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_strings.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_strings_internal.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_int128.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_throw_delegate.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_raw_logging_internal.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_severity.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_internal_message.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_internal_check_op.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_internal_nullguard.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_internal_format.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_internal_globals.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_internal_log_sink_set.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_globals.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_sink.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_str_format_internal.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_strerror.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_time.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_time_zone.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_civil_time.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_examine_stack.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_stacktrace.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_symbolize.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_debugging_internal.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_demangle_internal.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_malloc_internal.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_synchronization.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_graphcycles_internal.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_kernel_timeout_internal.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_internal_proto.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_log_internal_structured_proto.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_leak_check.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_city.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_hash.lib
  ${CESIUM_VCPKG_DIR}/lib/absl_low_level_hash.lib

)

target_link_libraries(csp-cesium-renderer PRIVATE
  ws2_32 crypt32 wldap32 normaliz bcrypt userenv iphlpapi secur32
)

target_compile_definitions(csp-cesium-renderer PRIVATE
  GLM_FORCE_XYZW_ONLY
  GLM_FORCE_EXPLICIT_CTOR
  GLM_FORCE_INTRINSICS
  GLM_ENABLE_EXPERIMENTAL
)

if(MSVC)
  target_compile_options(csp-cesium-renderer PRIVATE /utf-8 /Zc:__cplusplus /bigobj)
endif()

set_target_properties(csp-cesium-renderer PROPERTIES UNITY_BUILD OFF)

set_target_properties(csp-cesium-renderer PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
