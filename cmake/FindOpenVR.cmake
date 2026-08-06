# Locate header.
find_path(OPENVR_INCLUDE_DIR openvr.h
    HINTS ${OPENVR_ROOT_DIR}/include)

# Locate libraries.
find_library(OPENVR_LIBRARY NAMES openvr_api openvr_api.lib
    HINTS ${OPENVR_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenVR DEFAULT_MSG OPENVR_INCLUDE_DIR OPENVR_LIBRARY)

if (OPENVR_FOUND)
  set(OPENVR_INCLUDE_DIRS "${OPENVR_INCLUDE_DIR}")

  if (NOT OPENVR_FIND_QUIETLY)
    message(STATUS "OPENVR_INCLUDE_DIRS ............. ${OPENVR_INCLUDE_DIR}")
    message(STATUS "OPENVR_LIBRARY .................. ${OPENVR_LIBRARY}")
  endif ()

  if (NOT TARGET OpenVR::API)
    add_library(OpenVR::API UNKNOWN IMPORTED)
    set_target_properties(OpenVR::API PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OPENVR_INCLUDE_DIRS}")

    set_property(TARGET OpenVR::API APPEND PROPERTY
        IMPORTED_LOCATION "${OPENVR_LIBRARY}")
  endif ()
endif ()