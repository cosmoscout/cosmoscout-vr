# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #


# assets -------------------------------------------------------------------------------------------

install(DIRECTORY "models"    DESTINATION "share/resources")
install(DIRECTORY "textures"  DESTINATION "share/resources")
install(DIRECTORY "gui"       DESTINATION "share/resources")
install(DIRECTORY "icons"     DESTINATION "share/resources")

# chromium embedded framework ----------------------------------------------------------------------

if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
    set(CEF_RESOURCES
        "${CEF_LIBRARY_DIR}/natives_blob.bin"
        "${CEF_LIBRARY_DIR}/snapshot_blob.bin"
        "${CEF_LIBRARY_DIR}/v8_context_snapshot.bin"
        "${CEF_LIBRARY_DIR}/libcef.dll"
        "${CEF_LIBRARY_DIR}/chrome_elf.dll"
    )
    install(FILES ${CEF_RESOURCES} DESTINATION "bin")
    install(DIRECTORY "${CEF_LIBRARY_DIR}/swiftshader" DESTINATION "bin")
    install(DIRECTORY "${CEF_RESOURCE_DIR}/" DESTINATION "bin")
else()
    set(CEF_RESOURCES
        "${CEF_LIBRARY_DIR}/natives_blob.bin"
        "${CEF_LIBRARY_DIR}/snapshot_blob.bin"
        "${CEF_LIBRARY_DIR}/v8_context_snapshot.bin"
    )
    install(FILES ${CEF_RESOURCES} DESTINATION "lib")
    install(DIRECTORY "${CEF_LIBRARY_DIR}/swiftshader/" DESTINATION "lib")
    install(DIRECTORY "${CEF_RESOURCE_DIR}/" DESTINATION "lib" PATTERN "locales" EXCLUDE)
    install(DIRECTORY "${CEF_RESOURCE_DIR}/locales" DESTINATION "bin")
endif()

