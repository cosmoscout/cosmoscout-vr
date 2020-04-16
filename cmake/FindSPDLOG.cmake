# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# Locate header.
find_path(SPDLOG_INCLUDE_DIR spdlog/spdlog.h
    HINTS ${SPDLOG_ROOT_DIR}/include)

# Locate library.
find_library(SPDLOG_LIBRARY NAMES spdlog spdlogd
    HINTS ${SPDLOG_ROOT_DIR}/lib ${SPDLOG_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SPDLOG DEFAULT_MSG SPDLOG_INCLUDE_DIR SPDLOG_LIBRARY)

# Add imported target.
if(SPDLOG_FOUND)
    set(SPDLOG_INCLUDE_DIRS "${SPDLOG_INCLUDE_DIR}")

    if(NOT SPDLOG_FIND_QUIETLY)
        message(STATUS "SPDLOG_INCLUDE_DIRS ........... ${SPDLOG_INCLUDE_DIR}")
        message(STATUS "SPDLOG_LIBRARY ................ ${SPDLOG_LIBRARY}")
    endif()

    if(NOT TARGET spdlog::spdlog)
        add_library(spdlog::spdlog UNKNOWN IMPORTED)
        set_target_properties(spdlog::spdlog PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${SPDLOG_INCLUDE_DIRS}")

        set_property(TARGET spdlog::spdlog APPEND PROPERTY
            IMPORTED_LOCATION "${SPDLOG_LIBRARY}")
    endif()
endif()
