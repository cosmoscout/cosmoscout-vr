vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO cosmoscout/vista
    REF 6f0256af370ddd8e77f8467727ab09220968b83a
    SHA512 b00c4d536145ba940ad1053f92bfec22fcaed84283b2c96b86af92670ec68c9ab783cfef6c69a57bb830b8d470459bab518e7c19f55e6602a0539527e6371bc6
    HEAD_REF feature/vcpkg
)

vcpkg_check_features(
    OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
    window-sdl2              VISTA_CORELIBS_USE_SDL2_WINDOWIMP
    window-glut              VISTA_CORELIBS_USE_GLUT_WINDOWIMP
    openvr                   VISTA_CORELIBS_USE_OPENVR
    networking-zmq           VISTA_CORELIBS_USE_ZEROMQ
    openvr                   VISTA_DRIVERS_OPENVR
    driver-openal-record     VISTA_DRIVERS_OPENALRECORD
    driver-opencv-capture    VISTA_DRIVERS_OPENCVCAPTURE
    driver-opencv-face-track VISTA_DRIVERS_OPENCVFACETRACK
    driver-openni            VISTA_DRIVERS_OPENNI
    driver-wiimote           VISTA_DRIVERS_WIIMOTE
    driver-directx-gamepad   VISTA_DRIVERS_DIRECTXGAMEPAD
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
    ${FEATURE_OPTIONS}
    -DVISTA_DEMO=Off
    -DVISTA_TESTS=Off
    -DVISTA_ENABLE_WARNINGS=Off
    -DVISTA_CORELIBS_USE_INFINITE_REVERSE_PROJECTION=On
    -DVISTA_USE_PRECOMPILED_HEADERS=On
    -DCMAKE_UNITY_BUILD=On
    MAYBE_UNUSED_VARIABLES
    VISTA_DRIVERS_DIRECTXGAMEPAD
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
# vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")
#
# file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
# file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/share")
# file(INSTALL "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")