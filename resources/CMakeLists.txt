# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT


# assets -------------------------------------------------------------------------------------------

install(DIRECTORY "textures"  DESTINATION "share/resources")
install(DIRECTORY "gui"       DESTINATION "share/resources")
install(DIRECTORY "icons"     DESTINATION "share/resources")
install(DIRECTORY "locations" DESTINATION "share/resources")
install(DIRECTORY "shaders"   DESTINATION "share/resources")

# chromium embedded framework ----------------------------------------------------------------------

if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CEF_RESOURCES
        "${CEF_BINARY_DIR}/v8_context_snapshot.bin"
        "${CEF_BINARY_DIR}/libcef.dll"
        "${CEF_BINARY_DIR}/chrome_elf.dll"
        "${CEF_BINARY_DIR}/libEGL.dll"
        "${CEF_BINARY_DIR}/libGLESv2.dll"
    )
    install(FILES ${CEF_RESOURCES} DESTINATION "bin")
    install(DIRECTORY "${CEF_RESOURCE_DIR}/" DESTINATION "bin")
else()
    set(CEF_RESOURCES
        "${CEF_BINARY_DIR}/v8_context_snapshot.bin"
        "${CEF_BINARY_DIR}/libEGL.so"
        "${CEF_BINARY_DIR}/libGLESv2.so"
        "${CEF_BINARY_DIR}/libvk_swiftshader.so"
        "${CEF_BINARY_DIR}/libvulkan.so.1"
        "${CEF_BINARY_DIR}/vk_swiftshader_icd.json"
    )
    install(FILES ${CEF_RESOURCES} DESTINATION "lib")
    install(DIRECTORY "${CEF_RESOURCE_DIR}/" DESTINATION "lib" PATTERN "locales" EXCLUDE)
    install(DIRECTORY "${CEF_RESOURCE_DIR}/locales" DESTINATION "bin")
endif()

