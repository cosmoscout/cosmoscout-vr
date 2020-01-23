# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

# install platform dependent files -----------------------------------------------------------------
if (WIN32)
  install(FILES "spice/simple-windows.txt"  DESTINATION "share/config/spice" RENAME "simple.txt")
  install(FILES "scene/simple_vive.json"    DESTINATION "share/config")
  install(FILES "scripts/start.bat"         DESTINATION "bin")

  if (COSMOSCOUT_UNIT_TESTS)
    install(FILES "scripts/run_tests.bat" DESTINATION "bin")
  endif()
endif()

if (UNIX)
  install(FILES "spice/simple-linux.txt"   DESTINATION "share/config/spice" RENAME "simple.txt")
  install(FILES "scripts/start.sh"         DESTINATION "bin"
          PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
  
  if (COSMOSCOUT_UNIT_TESTS)
    install(FILES "scripts/run_tests.sh" DESTINATION "bin"
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
    install(FILES "scripts/run_graphical_tests.sh" DESTINATION "bin"
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ)
  endif()
endif()

# install files for all platforms ------------------------------------------------------------------
install(FILES "scene/simple_desktop.json" DESTINATION "share/config")
install(DIRECTORY "vista"                 DESTINATION "share/config")
