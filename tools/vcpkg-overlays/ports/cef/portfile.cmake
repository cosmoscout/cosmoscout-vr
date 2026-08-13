# =============================================================================
# CEF ships two things that need very different handling:
#   1. libcef.dll / libcef.lib and the rest of the Chromium runtime - these are
#      prebuilt by upstream and are NOT compiled here (there is no source for
#      them in the binary distribution).
#   2. libcef_dll_wrapper - a small static glue library whose source IS bundled
#      in the distribution, and which we build with the current triplet's
#      compiler/flags so it is ABI-compatible with the rest of the user's app.
#
# This mirrors upstream's own recommended usage (see cef_binary/CMakeLists.txt)
# and is why the vcpkg team never accepted a CEF port upstream: the curated
# microsoft/vcpkg registry requires ports build (nearly) everything from
# source. See https://github.com/microsoft/vcpkg/pull/10102 for the history.
# This port is intended as a private overlay port, not for upstream vcpkg.
# =============================================================================

vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

if(VCPKG_TARGET_IS_WINDOWS AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  # Windows‑x64 is supported – nothing to do here
elseif(VCPKG_TARGET_IS_LINUX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  # Linux (x64 or arm64) is supported – nothing to do here
else()
  message(FATAL_ERROR
      "The cef port currently supports only:\n"
      "  - windows-x64\n"
      "  - linux-x64\n"
      "Unsupported triplet: ${VCPKG_TARGET_TRIPLET}")
endif()

# The "minimal" distribution only ships Release binaries, so there is nothing
# to gain from (and no way to satisfy) a Debug build here.
set(VCPKG_BUILD_TYPE release)

set(CEF_VERSION "135.0.20+ge7de5c3+chromium-135.0.7049.85")

set(CEF_PLATFORM "unsupported")
if(VCPKG_TARGET_IS_WINDOWS AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  set(CEF_PLATFORM "windows64")
elseif(VCPKG_TARGET_IS_LINUX AND VCPKG_TARGET_ARCHITECTURE STREQUAL "x64")
  set(CEF_PLATFORM "linux64")
endif()

set(CEF_DISTRIB_TYPE "minimal")
set(CEF_ARCHIVE_NAME "cef_binary_${CEF_VERSION}_${CEF_PLATFORM}_${CEF_DISTRIB_TYPE}")

# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
# The SHA512 below is a placeholder. Run the install once with it left as-is;
# vcpkg will refuse to proceed and print the actual hash of the downloaded
# file, which you then paste in here. This is the standard way to pin a
# distfile's hash without fetching it out-of-band.

if (VCPKG_TARGET_IS_WINDOWS)
  set(CEF_SHA512 17edb65628c7d1f82a91a156d6ea3f01420f4ef97fa8b1d7c7072f2973ee0ec7a20121e2cbb4760368f9978a55aac4a6c6f3a38ab239417a98cbe2170ce82e8d)
elseif (VCPKG_TARGET_IS_LINUX)
  set(CEF_SHA512 481bac3c124070c79cff0d9175d22f36233f29d4886b1e1bf0127e2ea11ecefbae838c5d04b90b9e4a06b085abe5ba9d787f9af7be2351c99f3ee21e40c7645a)
endif ()

vcpkg_download_distfile(ARCHIVE
    URLS "https://cef-builds.spotifycdn.com/${CEF_ARCHIVE_NAME}.tar.bz2"
    FILENAME "${CEF_ARCHIVE_NAME}.tar.bz2"
    SHA512 ${CEF_SHA512}
)

vcpkg_extract_source_archive_ex(
    OUT_SOURCE_PATH SOURCE_PATH
    ARCHIVE "${ARCHIVE}"
    REF "${CEF_VERSION}"
)

# Defensive: not present in every "minimal" build, but strip it if it is.
if(EXISTS "${SOURCE_PATH}/tests")
  file(REMOVE_RECURSE "${SOURCE_PATH}/tests")
endif()

# vcpkg's toolchain appends -DBUILD_SHARED_LIBS=ON (based on VCPKG_LIBRARY_LINKAGE)
# after the portfile's OPTIONS, so cmake sees both flags and the toolchain's
# value (the last one) wins. Rather than fight the flag order, we patch the
# libcef_dll CMakeLists.txt to use explicit STATIC so BUILD_SHARED_LIBS has no
# effect on the wrapper target. This mirrors what the portfile's comments already
# describe as the intent.
vcpkg_replace_string(
    "${SOURCE_PATH}/libcef_dll/CMakeLists.txt"
    "add_library(\${CEF_TARGET}"
    "add_library(\${CEF_TARGET} STATIC"
)

# ----------------------------------------------------------------------------
# Configure + build libcef_dll_wrapper only
# ----------------------------------------------------------------------------
# vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY) above guarantees VCPKG_CRT_LINKAGE
# is "dynamic" for every triplet that reaches this point, but we derive the
# flag anyway rather than hardcoding /MD, in case that constraint ever loosens.
set(CEF_CONFIGURE_OPTIONS
    -DUSE_SANDBOX=OFF
    # Document intent (vcpkg's toolchain overrides this to ON at the end
    # of the cmake command line anyway - see the patch below for the real
    # fix). The wrapper must always be static: it's glue code that gets
    # linked into your app and resolves its libcef symbols against
    # libcef.lib/.dll at your app's link/load time, not its own.  Without
    # the STATIC patch the wrapper is built as a DLL and fails to link with
    # "unresolved external symbol cef_string_utf16_set" (or any other
    # libcef-exported symbol it references), since nothing has linked
    # libcef.lib into it at that point.
    -DBUILD_SHARED_LIBS=OFF
)

if(VCPKG_TARGET_IS_WINDOWS)
  if(VCPKG_CRT_LINKAGE STREQUAL "dynamic")
    set(CEF_RUNTIME_LIBRARY_FLAG "/MD")
  else()
    set(CEF_RUNTIME_LIBRARY_FLAG "/MT")
  endif()

  list(APPEND CEF_CONFIGURE_OPTIONS
      -DCEF_RUNTIME_LIBRARY_FLAG=${CEF_RUNTIME_LIBRARY_FLAG}
      -DCEF_DEBUG_INFO_FLAG=
  )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
    ${CEF_CONFIGURE_OPTIONS}
)

vcpkg_cmake_build(TARGET libcef_dll_wrapper)

# ----------------------------------------------------------------------------
# Install
# ----------------------------------------------------------------------------
# CEF's own CMakeLists.txt has no install() rules at all for the binary
# distribution - upstream expects you to copy files manually (exactly what
# your batch script did), so we replicate that step by hand instead of calling
# vcpkg_cmake_install().

# Headers: CEF's own headers use the convention #include "include/cef_xxx.h",
# i.e. they expect the *parent* of "include/" on the include path, not
# "include/" itself. Keeping a "cef/include/..." nesting here lets the
# cef-config.cmake set INTERFACE_INCLUDE_DIRECTORIES to .../include/cef and
# have consumer code's #include "include/cef_app.h" resolve correctly.
file(COPY "${SOURCE_PATH}/include" DESTINATION "${CURRENT_PACKAGES_DIR}/include/cef")

# Prebuilt Release binaries (libcef.dll and the rest of the Chromium runtime).
file(GLOB CEF_RELEASE_FILES "${SOURCE_PATH}/Release/*")
file(COPY ${CEF_RELEASE_FILES} DESTINATION "${CURRENT_PACKAGES_DIR}/bin")

# Import libraries (.lib) belong in lib/, not bin/.
file(GLOB CEF_IMPORT_LIBS "${CURRENT_PACKAGES_DIR}/bin/*.lib")
if(CEF_IMPORT_LIBS)
  file(COPY ${CEF_IMPORT_LIBS} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")
  file(REMOVE ${CEF_IMPORT_LIBS})
endif()

# The wrapper we just built. Search recursively since the exact path differs
# between single-config (Ninja) and multi-config (MSBuild) generators.
# On Windows the wrapper is a .lib file, while on Linux it is a static .a archive.
# Use a glob that matches both extensions to handle both platforms.
file(GLOB_RECURSE CEF_WRAPPER_LIB
    "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/*libcef_dll_wrapper.lib"
    "${CURRENT_BUILDTREES_DIR}/${TARGET_TRIPLET}-rel/*libcef_dll_wrapper.a")
list(LENGTH CEF_WRAPPER_LIB CEF_WRAPPER_LIB_COUNT)
if(NOT CEF_WRAPPER_LIB_COUNT EQUAL 1)
  message(FATAL_ERROR "Expected exactly one libcef_dll_wrapper library, found: ${CEF_WRAPPER_LIB}")
endif()
file(COPY ${CEF_WRAPPER_LIB} DESTINATION "${CURRENT_PACKAGES_DIR}/lib")

# Runtime resources: *.pak files and locales/.
file(COPY "${SOURCE_PATH}/Resources/" DESTINATION "${CURRENT_PACKAGES_DIR}/share/cef/Resources")

# find_package(cef CONFIG) support + a helper to copy runtime files next to
# a consumer's executable.
file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/cef-config.cmake" DESTINATION "${CURRENT_PACKAGES_DIR}/share/cef")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")

# ----------------------------------------------------------------------------
# Post-build policy exceptions
# ----------------------------------------------------------------------------
# Several of CEF's redistributable DLLs (d3dcompiler_47.dll, the SwiftShader
# and ANGLE/EGL DLLs, dxil.dll, dxcompiler.dll, ...) are loaded dynamically at
# runtime and intentionally ship without a matching .lib import library. This
# is expected for this port, not a packaging mistake.
set(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)
set(VCPKG_POLICY_DLLS_WITHOUT_EXPORTS enabled)
set(VCPKG_POLICY_MISMATCHED_NUMBER_OF_BINARIES enabled)